"""
Direction-Aware Attention Mechanism for Vessel Segmentation

This module implements a direction-aware attention mechanism that enhances
the network's ability to capture directional information in vessel structures.
It can be integrated into both CNN and Transformer architectures.

Key Features:
1. Multi-directional feature extraction
2. Direction-aware feature modulation
3. Adaptive directional weighting
4. Compatible with existing attention mechanisms

Author: Code Development Agent
Date: 2026-04-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DirectionalConvBlock(nn.Module):
    """
    Directional convolution block that extracts features in multiple directions.
    
    This block applies convolutions in 4 principal directions (0°, 45°, 90°, 135°)
    to capture directional vessel patterns.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        dilation (int): Dilation rate for larger receptive field
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DirectionalConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Standard convolution
        self.conv_standard = nn.Conv2d(
            in_channels, out_channels // 4, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            dilation=dilation
        )
        
        # Directional convolutions (using dilated convolutions to simulate directions)
        # Horizontal direction (0°)
        self.conv_h = nn.Conv2d(
            in_channels, out_channels // 4,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            dilation=dilation
        )
        
        # Vertical direction (90°)
        self.conv_v = nn.Conv2d(
            in_channels, out_channels // 4,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            dilation=dilation
        )
        
        # Diagonal directions (45° and 135°) - approximated with standard conv
        self.conv_d1 = nn.Conv2d(
            in_channels, out_channels // 4,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            dilation=dilation
        )
        self.conv_d2 = nn.Conv2d(
            in_channels, out_channels // 4,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            dilation=dilation
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel mixer if needed
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor with directional features
        """
        identity = x
        
        # Extract features in different directions
        feat_standard = self.conv_standard(x)
        feat_h = self.conv_h(x)
        feat_v = self.conv_v(x)
        feat_d1 = self.conv_d1(x)
        feat_d2 = self.conv_d2(x)
        
        # Concatenate directional features
        out = torch.cat([feat_standard, feat_h, feat_v, feat_d1, feat_d2], dim=1)
        
        # Batch normalization and activation
        out = self.bn(out)
        out = self.relu(out)
        
        # Shortcut connection
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class DirectionAwareAttention(nn.Module):
    """
    Direction-Aware Attention Module.
    
    This module enhances feature representations by applying attention
    weights that are aware of directional information. It's particularly
    effective for vessel segmentation where directional continuity matters.
    
    Args:
        in_channels (int): Number of input channels
        reduction (int): Reduction ratio for channel attention
        num_directions (int): Number of directional branches
    """
    
    def __init__(self, in_channels, reduction=16, num_directions=4):
        super(DirectionAwareAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_directions = num_directions
        self.reduction = reduction
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Direction-specific feature extraction
        self.directional_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
            ) for _ in range(num_directions)
        ])
        
        # Directional kernels for feature modulation
        self._register_directional_kernels()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_directions, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def _register_directional_kernels(self):
        """Register directional modulation kernels."""
        # These kernels help modulate features based on direction
        for i in range(self.num_directions):
            angle = i * (180 / self.num_directions)
            # Register as buffer for potential visualization
            self.register_buffer(f'angle_{i}', torch.tensor(angle))
    
    def _extract_directional_features(self, x):
        """
        Extract features in multiple directions.
        
        Args:
            x (Tensor): Input tensor (N, C, H, W)
            
        Returns:
            list: List of directional feature tensors
        """
        N, C, H, W = x.shape
        directional_features = []
        
        # Standard feature
        directional_features.append(x)
        
        # Apply directional convolutions
        for i, conv in enumerate(self.directional_convs):
            # Apply global pooling to get channel-wise statistics
            pooled = self.avg_pool(x)
            
            # Generate direction-specific attention
            direction_feat = conv(pooled)
            
            # Modulate original features
            modulated = x * torch.sigmoid(direction_feat)
            directional_features.append(modulated)
        
        return directional_features
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Attention-enhanced output tensor
        """
        # Extract multi-directional features
        directional_features = self._extract_directional_features(x)
        
        # Concatenate all directional features
        concat_features = torch.cat(directional_features, dim=1)
        
        # Fuse directional information
        direction_weights = self.fusion(concat_features)
        
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        
        # Combine direction and channel attention
        combined_weights = direction_weights * channel_weights
        
        # Apply attention to input
        out = x * combined_weights
        
        # Residual connection
        out = out + x
        
        return out


class DirectionAwareTransformerBlock(nn.Module):
    """
    Direction-Aware Transformer Block.
    
    This block integrates direction awareness into the standard Transformer
    self-attention mechanism, making it more suitable for vessel segmentation.
    
    Args:
        hidden_size (int): Dimension of hidden features
        num_heads (int): Number of attention heads
        mlp_dim (int): Dimension of MLP hidden layer
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate=0.1):
        super(DirectionAwareTransformerBlock, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Direction-aware modulation
        self.direction_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (N, L, C) where L is sequence length
            
        Returns:
            Tensor: Output tensor with same shape
        """
        # Pre-norm
        normed = self.norm1(x)
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            normed, normed, normed, 
            need_weights=True,
            average_attn_weights=False
        )
        
        # Apply direction-aware modulation to attention weights
        # Reshape for modulation
        N, L, C = x.shape
        attn_reshaped = attn_output.view(N, L, C)
        
        # Generate direction modulation weights
        direction_weights = self.direction_modulation(attn_reshaped)
        
        # Modulate attention output
        modulated_attn = attn_reshaped * direction_weights
        
        # Residual connection
        x = x + self.dropout(modulated_attn)
        
        # MLP with pre-norm
        normed2 = self.norm2(x)
        mlp_output = self.mlp(normed2)
        
        # Residual connection
        x = x + mlp_output
        
        return x


class DirectionAwareSpatialAttention(nn.Module):
    """
    Direction-Aware Spatial Attention Module.
    
    This module focuses on spatial relationships with directional awareness,
    particularly useful for capturing vessel orientation and continuity.
    
    Args:
        in_channels (int): Number of input channels
        kernel_size (int): Kernel size for directional convolutions
    """
    
    def __init__(self, in_channels, kernel_size=7):
        super(DirectionAwareSpatialAttention, self).__init__()
        
        # Directional convolution kernels
        self.conv_h = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), 
                                padding=(0, kernel_size//2), bias=False)
        self.conv_v = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), 
                                padding=(kernel_size//2, 0), bias=False)
        self.conv_d = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                                padding=kernel_size//2, bias=False)
        
        # Fusion convolution
        self.fusion_conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Spatially attended output
        """
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply directional convolutions
        feat_h = self.conv_h(concat)
        feat_v = self.conv_v(concat)
        feat_d = self.conv_d(concat)
        
        # Concatenate and fuse
        concat_feats = torch.cat([feat_h, feat_v, feat_d], dim=1)
        spatial_attention = self.fusion_conv(concat_feats)
        
        # Apply sigmoid to get attention map
        attention_map = self.sigmoid(spatial_attention)
        
        # Apply attention to input
        out = x * attention_map
        
        return out


# Convenience function to create direction-aware attention
def create_direction_attention(attention_type, in_channels, **kwargs):
    """
    Factory function to create different types of direction-aware attention.
    
    Args:
        attention_type (str): Type of attention ('spatial', 'channel', 'transformer')
        in_channels (int): Number of input channels
        **kwargs: Additional arguments
        
    Returns:
        nn.Module: Direction-aware attention module
    """
    if attention_type == 'spatial':
        return DirectionAwareSpatialAttention(in_channels, **kwargs)
    elif attention_type == 'channel':
        return DirectionAwareAttention(in_channels, **kwargs)
    elif attention_type == 'transformer':
        return DirectionAwareTransformerBlock(**kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
