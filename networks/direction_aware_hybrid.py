"""
Direction-Aware Hybrid Mamba-Transformer for Vessel Segmentation

Architecture:
Direction-Aware CNN Encoder → [Direction-Aware Mamba + Direction-Aware Transformer] → Direction-Aware CNN Decoder

Key Features:
1. Dual-path processing: Mamba for efficient local-medium range, Transformer for global
2. Direction-aware throughout the network
3. Parallel or sequential fusion of Mamba and Transformer features
4. Optimized for vessel segmentation with direction priors

Author: Code Development Agent (Advanced Mamba Integration)
Date: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

# Import Mamba modules
try:
    from .mamba_block import SimpleMambaBlock, MambaConfig, MambaEncoder
except ImportError:
    from mamba_block import SimpleMambaBlock, MambaConfig, MambaEncoder


class DirectionEncoder(nn.Module):
    """
    Encodes direction information into feature embeddings.
    
    Args:
        num_directions: Number of direction bins (default: 4 for 0°, 45°, 90°, 135°)
        d_model: Output dimension
    """
    
    def __init__(self, num_directions: int = 4, d_model: int = 256):
        super().__init__()
        self.num_directions = num_directions
        
        self.direction_encoder = nn.Sequential(
            nn.Linear(num_directions, d_model // 4),
            nn.GELU(),
            nn.LayerNorm(d_model // 4),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, direction_features: torch.Tensor) -> torch.Tensor:
        """
        Encode direction features.
        
        Args:
            direction_features: (batch, num_directions) or (batch, seq_len, num_directions)
            
        Returns:
            direction_embedding: (batch, d_model) or (batch, seq_len, d_model)
        """
        if direction_features.dim() == 2:
            return self.direction_encoder(direction_features)
        elif direction_features.dim() == 3:
            B, L, _ = direction_features.shape
            direction_features = rearrange(direction_features, 'b l d -> (b l) d')
            emb = self.direction_encoder(direction_features)
            return rearrange(emb, '(b l) d -> b l d', b=B, l=L)
        else:
            raise ValueError(f"Invalid direction_features shape: {direction_features.shape}")


class DirectionAwareMamba(nn.Module):
    """
    Mamba block with direction-aware modulation.
    
    Args:
        d_model: Model dimension
        d_state: State dimension
        num_directions: Number of direction bins
        modulation_type: 'gate' or 'add' or 'multiply'
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_directions: int = 4,
        modulation_type: str = 'gate',
    ):
        super().__init__()
        self.d_model = d_model
        self.num_directions = num_directions
        self.modulation_type = modulation_type
        
        # Direction encoding
        self.direction_encoder = DirectionEncoder(num_directions, d_model)
        
        # Mamba block
        self.mamba = SimpleMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        
        # Direction modulation
        if modulation_type == 'gate':
            self.direction_modulation = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        elif modulation_type == 'add':
            self.direction_modulation = nn.Linear(d_model, d_model)
        elif modulation_type == 'multiply':
            self.direction_modulation = nn.Linear(d_model, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, direction_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with direction modulation.
        
        Args:
            x: Input (batch, seq_len, d_model)
            direction_features: Optional (batch, num_directions) or (batch, seq_len, num_directions)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        
        # Apply direction modulation
        if direction_features is not None:
            dir_embed = self.direction_encoder(direction_features)
            
            # Ensure correct shape for modulation
            if dir_embed.dim() == 2:
                # (batch, d_model) -> (batch, 1, d_model)
                dir_embed = dir_embed.unsqueeze(1)
            
            if self.modulation_type == 'gate':
                modulation = self.direction_modulation(dir_embed)
                x = x * modulation
            elif self.modulation_type == 'add':
                modulation = self.direction_modulation(dir_embed)
                x = x + modulation
            elif self.modulation_type == 'multiply':
                modulation = self.direction_modulation(dir_embed)
                x = x * (1 + modulation)
        
        # Mamba processing
        x = self.mamba(x)
        
        return x + residual


class DirectionAwareTransformer(nn.Module):
    """
    Transformer block with direction-aware modulation.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_directions: Number of direction bins
        modulation_type: 'gate' or 'add' or 'multiply'
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_directions: int = 4,
        modulation_type: str = 'gate',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_directions = num_directions
        
        # Direction encoding
        self.direction_encoder = DirectionEncoder(num_directions, d_model)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        # Direction modulation for attention
        if modulation_type == 'gate':
            self.direction_modulation = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        elif modulation_type == 'add':
            self.direction_modulation = nn.Linear(d_model, d_model)
        elif modulation_type == 'multiply':
            self.direction_modulation = nn.Linear(d_model, d_model)
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, direction_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with direction modulation.
        
        Args:
            x: Input (batch, seq_len, d_model)
            direction_features: Optional (batch, num_directions) or (batch, seq_len, num_directions)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        # Self-attention with direction modulation
        residual = x
        x = self.norm1(x)
        
        if direction_features is not None:
            dir_embed = self.direction_encoder(direction_features)
            
            # Ensure correct shape for modulation
            if dir_embed.dim() == 2:
                dir_embed = dir_embed.unsqueeze(1)
            
            modulation = self.direction_modulation(dir_embed)
            x = x * modulation
        
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = residual + self.dropout(attn_output)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


class HybridMambaTransformerBlock(nn.Module):
    """
    Hybrid block combining Direction-Aware Mamba and Direction-Aware Transformer.
    
    Supports parallel or sequential fusion.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_state: Mamba state dimension
        num_directions: Number of direction bins
        fusion_type: 'parallel' or 'sequential' or 'gated'
        sequential_order: 'mamba_first' or 'transformer_first'
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_state: int = 16,
        num_directions: int = 4,
        fusion_type: str = 'parallel',
        sequential_order: str = 'mamba_first',
        modulation_type: str = 'gate',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.sequential_order = sequential_order
        
        # Direction-Aware Mamba
        self.direction_mamba = DirectionAwareMamba(
            d_model=d_model,
            d_state=d_state,
            num_directions=num_directions,
            modulation_type=modulation_type,
        )
        
        # Direction-Aware Transformer
        self.direction_transformer = DirectionAwareTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_directions=num_directions,
            modulation_type=modulation_type,
            dropout=dropout,
        )
        
        # Fusion mechanisms
        if fusion_type == 'parallel':
            # Parallel: combine outputs from both branches
            self.fusion = nn.Linear(d_model * 2, d_model)
        elif fusion_type == 'sequential':
            # Sequential: Mamba → Transformer or Transformer → Mamba
            pass  # No additional parameters needed
        elif fusion_type == 'gated':
            # Gated fusion: learnable gate to combine
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        direction_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hybrid block.
        
        Args:
            x: Input (batch, seq_len, d_model)
            direction_features: Optional (batch, num_directions)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        if self.fusion_type == 'parallel':
            # Parallel processing
            mamba_out = self.direction_mamba(x, direction_features)
            transformer_out = self.direction_transformer(x, direction_features)
            
            # Concatenate and fuse
            combined = torch.cat([mamba_out, transformer_out], dim=-1)
            out = self.fusion(combined)
            
            return out
        
        elif self.fusion_type == 'sequential':
            # Sequential processing
            if self.sequential_order == 'mamba_first':
                x = self.direction_mamba(x, direction_features)
                x = self.direction_transformer(x, direction_features)
            else:  # transformer_first
                x = self.direction_transformer(x, direction_features)
                x = self.direction_mamba(x, direction_features)
            
            return x
        
        elif self.fusion_type == 'gated':
            # Gated fusion
            mamba_out = self.direction_mamba(x, direction_features)
            transformer_out = self.direction_transformer(x, direction_features)
            
            combined = torch.cat([mamba_out, transformer_out], dim=-1)
            gate = self.gate(combined)
            
            out = gate * mamba_out + (1 - gate) * transformer_out
            
            return out


class DirectionAwareHybridEncoder(nn.Module):
    """
    Multi-layer Direction-Aware Hybrid Mamba-Transformer Encoder.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of hybrid blocks
        d_state: Mamba state dimension
        num_directions: Number of direction bins
        fusion_type: 'parallel' or 'sequential' or 'gated'
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_state: int = 16,
        num_directions: int = 4,
        fusion_type: str = 'parallel',
        modulation_type: str = 'gate',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Hybrid blocks
        self.blocks = nn.ModuleList([
            HybridMambaTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_state=d_state,
                num_directions=num_directions,
                fusion_type=fusion_type,
                modulation_type=modulation_type,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        direction_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input (batch, seq_len, d_model)
            direction_features: Optional (batch, num_directions)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x, direction_features)
        
        return self.norm(x)


class DirectionAwareHybridBottleneck(nn.Module):
    """
    Complete bottleneck module for vessel segmentation.
    
    Architecture:
    CNN features → Patch Embedding → Direction-Aware Hybrid Encoder → Projection → CNN features
    
    Args:
        in_channels: Input channels from CNN encoder
        out_channels: Output channels to CNN decoder
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of hybrid blocks
        patch_size: Patch size for embedding
        num_directions: Number of direction bins
        fusion_type: 'parallel' or 'sequential' or 'gated'
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_state: int = 16,
        patch_size: int = 1,
        num_directions: int = 4,
        fusion_type: str = 'parallel',
        modulation_type: str = 'gate',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        # CNN to Transformer/Mamba projection
        self.cnn_to_transformer = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        
        # Direction encoder (from CNN features)
        self.direction_encoder = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 4, num_directions, kernel_size=1),
            nn.Softmax(dim=1),
        )
        
        # Hybrid encoder
        self.hybrid_encoder = DirectionAwareHybridEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_state=d_state,
            num_directions=num_directions,
            fusion_type=fusion_type,
            modulation_type=modulation_type,
            dropout=dropout,
        )
        
        # Transformer/Mamba to CNN projection
        self.transformer_to_cnn = nn.Sequential(
            nn.Conv2d(d_model, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottleneck.
        
        Args:
            x: Input (batch, in_channels, H, W)
            
        Returns:
            Output (batch, out_channels, H, W)
        """
        batch, _, H, W = x.shape
        
        # CNN to Transformer projection
        x_proj = self.cnn_to_transformer(x)  # (batch, d_model, H, W)
        
        # Extract direction features
        dir_features = self.direction_encoder(x)  # (batch, num_directions, H, W)
        dir_features = rearrange(dir_features, 'b d h w -> b (h w) d')
        
        # Flatten to sequence
        x_seq = rearrange(x_proj, 'b d h w -> b (h w) d')  # (batch, H*W, d_model)
        
        # Hybrid encoding
        x_encoded = self.hybrid_encoder(x_seq, dir_features)  # (batch, H*W, d_model)
        
        # Reshape back to image
        x_out = rearrange(x_encoded, 'b (h w) d -> b d h w', h=H, w=W)
        
        # Transformer to CNN projection
        x_out = self.transformer_to_cnn(x_out)
        
        # Skip connection
        skip = self.skip_conv(x)
        
        return x_out + skip


if __name__ == "__main__":
    # Test Direction-Aware Mamba
    B, L, D = 2, 196, 256
    x = torch.randn(B, L, D)
    dir_feat = torch.randn(B, 4)
    
    da_mamba = DirectionAwareMamba(d_model=D)
    y = da_mamba(x, dir_feat)
    print(f"DirectionAwareMamba: {x.shape} + {dir_feat.shape} -> {y.shape}")
    
    # Test Direction-Aware Transformer
    da_transformer = DirectionAwareTransformer(d_model=D)
    y = da_transformer(x, dir_feat)
    print(f"DirectionAwareTransformer: {x.shape} + {dir_feat.shape} -> {y.shape}")
    
    # Test Hybrid Block (Parallel)
    hybrid_parallel = HybridMambaTransformerBlock(d_model=D, fusion_type='parallel')
    y = hybrid_parallel(x, dir_feat)
    print(f"HybridBlock (Parallel): {x.shape} + {dir_feat.shape} -> {y.shape}")
    
    # Test Hybrid Block (Sequential)
    hybrid_sequential = HybridMambaTransformerBlock(d_model=D, fusion_type='sequential')
    y = hybrid_sequential(x, dir_feat)
    print(f"HybridBlock (Sequential): {x.shape} + {dir_feat.shape} -> {y.shape}")
    
    # Test Hybrid Block (Gated)
    hybrid_gated = HybridMambaTransformerBlock(d_model=D, fusion_type='gated')
    y = hybrid_gated(x, dir_feat)
    print(f"HybridBlock (Gated): {x.shape} + {dir_feat.shape} -> {y.shape}")
    
    # Test Hybrid Encoder
    encoder = DirectionAwareHybridEncoder(d_model=D, num_layers=2)
    y = encoder(x, dir_feat)
    print(f"DirectionAwareHybridEncoder: {x.shape} + {dir_feat.shape} -> {y.shape}")
    
    # Test Bottleneck
    B, C, H, W = 2, 512, 14, 14
    x_cnn = torch.randn(B, C, H, W)
    
    bottleneck = DirectionAwareHybridBottleneck(
        in_channels=C,
        out_channels=C,
        d_model=256,
        num_layers=2,
        fusion_type='parallel',
    )
    y = bottleneck(x_cnn)
    print(f"DirectionAwareHybridBottleneck: {x_cnn.shape} -> {y.shape}")
    
    print("\n✅ All Direction-Aware Hybrid Mamba-Transformer modules tested successfully!")
