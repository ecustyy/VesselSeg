"""
Simplified Mamba (State Space Model) Block for Vessel Segmentation

This is a simplified but functional implementation of Mamba-inspired SSM blocks
for vessel segmentation. It captures the key ideas:
1. State space modeling for long-range dependencies
2. Selective state updates
3. Linear complexity O(N)

For production use, consider the official Mamba package:
pip install mamba-ssm

Author: Code Development Agent (Mamba Integration)
Date: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class MambaConfig:
    """Configuration for Mamba blocks."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.num_layers = num_layers
        self.dropout = dropout


class SimpleMambaBlock(nn.Module):
    """
    Simplified Mamba-inspired SSM block.
    
    Uses a learnable state space with selective updates.
    More efficient than full Mamba but captures key benefits.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Local convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        
        # Projections
        self.x_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq_len, d_model)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = self.act(x_conv)
        
        # State space computation (simplified)
        x_ssm = self._state_space(x_conv, batch, seq_len)
        
        # Gate
        z = self.act(z)
        
        # Output
        out = x_ssm * z
        out = self.dropout(out)
        out = self.out_proj(out)
        
        return out
    
    def _state_space(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """
        Simplified state space computation.
        
        Implements a learnable recurrence that captures long-range dependencies.
        """
        # Project to state space
        B = self.x_proj(x)  # (batch, seq_len, d_state)
        
        # Compute delta (step size)
        dt = self.dt_proj(x)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)
        
        # Discretize state space (clamp dA directly for stability)
        dt_A = dt.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(torch.clamp(dt_A, -20.0, 20.0))
        dA = torch.clamp(dA, 1e-6, 1e6)  # (batch, seq_len, d_inner, d_state)
        
        # Initialize hidden state
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        # Sequential state update
        outputs = []
        for t in range(seq_len):
            # State update: h_t = dA_t * h_{t-1} + B_t * x_t
            h = dA[:, t] * h + B[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h = torch.clamp(h, -100.0, 100.0)  # numerical stability
            
            # Output: y_t = sum(h_t) + D * x_t
            y_t = h.sum(dim=-1) + self.D * x[:, t]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        return y


class MambaLayer(nn.Module):
    """Mamba layer with normalization and residual."""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = SimpleMambaBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dropout=config.dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual


class MambaEncoder(nn.Module):
    """Multi-layer Mamba encoder."""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.layers = nn.ModuleList([MambaLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class HybridMambaCNN(nn.Module):
    """
    Hybrid Mamba-CNN architecture for vessel segmentation.
    
    Uses CNN for local features and Mamba for long-range dependencies.
    Can be used as a drop-in replacement for Transformer bottleneck.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        d_model: Model dimension
        num_mamba_layers: Number of Mamba layers
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        d_model: int = 256,
        num_mamba_layers: int = 2,
    ):
        super().__init__()
        
        # CNN to Mamba projection
        self.cnn_to_mamba = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        
        # Mamba encoder
        mamba_config = MambaConfig(d_model=d_model, num_layers=num_mamba_layers)
        self.mamba_encoder = MambaEncoder(mamba_config)
        
        # Mamba to CNN projection
        self.mamba_to_cnn = nn.Sequential(
            nn.Conv2d(d_model, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch, in_channels, H, W)
            
        Returns:
            Output (batch, out_channels, H, W)
        """
        batch, _, H, W = x.shape
        
        # CNN to Mamba
        x = self.cnn_to_mamba(x)  # (batch, d_model, H, W)
        
        # Flatten to sequence
        x = rearrange(x, 'b d h w -> b (h w) d')  # (batch, H*W, d_model)
        
        # Mamba encoding
        x = self.mamba_encoder(x)  # (batch, H*W, d_model)
        
        # Reshape back to image
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        
        # Mamba to CNN
        x = self.mamba_to_cnn(x)
        
        return x


if __name__ == "__main__":
    # Test Mamba block
    B, L, D = 2, 196, 256
    x = torch.randn(B, L, D)
    
    # Test basic Mamba
    mamba = SimpleMambaBlock(d_model=D)
    y = mamba(x)
    print(f"SimpleMambaBlock: {x.shape} -> {y.shape}")
    
    # Test Mamba layer
    config = MambaConfig(d_model=D)
    layer = MambaLayer(config)
    y = layer(x)
    print(f"MambaLayer: {x.shape} -> {y.shape}")
    
    # Test Mamba encoder
    encoder = MambaEncoder(config)
    y = encoder(x)
    print(f"MambaEncoder: {x.shape} -> {y.shape}")
    
    # Test Hybrid Mamba-CNN
    B, C, H, W = 2, 512, 14, 14
    x_cnn = torch.randn(B, C, H, W)
    hybrid = HybridMambaCNN(in_channels=C, out_channels=C, d_model=256)
    y = hybrid(x_cnn)
    print(f"HybridMambaCNN: {x_cnn.shape} -> {y.shape}")
    
    print("\n✅ All Mamba modules tested successfully!")
