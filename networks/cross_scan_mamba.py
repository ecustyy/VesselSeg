"""
Cross-Scan Mamba with Direction-Specific Merge (DSM)
Based on VMamba 2D-Selective-Scan, with enhanced directional fusion.

Architecture:
  Input 2D features (B, C, H, W)
    -> CrossScan: 4-direction unroll (l2r, r2l, t2b, b2t)
    -> 4x S6 Block (shared params)
    -> CrossMerge: reshape back to 2D
    -> Direction-Specific Merge (DSM)
    -> Output (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
from networks.mamba_block import SimpleMambaBlock


class CrossScan(nn.Module):
    """Unroll 2D feature map into 4 directional 1D sequences."""

    def forward(self, x):
        B, C, H, W = x.shape
        x_l2r = rearrange(x, 'b c h w -> b (h w) c')
        x_r2l = torch.flip(x, dims=[3])
        x_r2l = rearrange(x_r2l, 'b c h w -> b (h w) c')
        x_t2b = rearrange(x, 'b c h w -> b c w h')
        x_t2b = rearrange(x_t2b, 'b c w h -> b (w h) c')
        x_b2t = rearrange(x, 'b c h w -> b c w h')
        x_b2t = torch.flip(x_b2t, dims=[3])
        x_b2t = rearrange(x_b2t, 'b c w h -> b (w h) c')
        return {'l2r': x_l2r, 'r2l': x_r2l, 't2b': x_t2b, 'b2t': x_b2t}, (B, C, H, W)


class CrossMerge(nn.Module):
    """Reshape 4 directional 1D sequences back to 2D."""

    def forward(self, seqs, shape):
        B, C, H, W = shape
        out = {}
        out['l2r'] = rearrange(seqs['l2r'], 'b (h w) c -> b c h w', h=H, w=W)
        f_r2l = rearrange(seqs['r2l'], 'b (h w) c -> b c h w', h=H, w=W)
        out['r2l'] = torch.flip(f_r2l, dims=[3])
        f_t2b = rearrange(seqs['t2b'], 'b (w h) c -> b c w h', w=W, h=H)
        out['t2b'] = rearrange(f_t2b, 'b c w h -> b c h w')
        f_b2t = rearrange(seqs['b2t'], 'b (w h) c -> b c w h', w=W, h=H)
        f_b2t = torch.flip(f_b2t, dims=[2])
        out['b2t'] = rearrange(f_b2t, 'b c w h -> b c h w')
        return out


class DirectionSpecificMerge(nn.Module):
    """
    Enhanced directional fusion: depthwise conv + inverted bottleneck MLP + channel gate.
    """
    def __init__(self, dim, expansion_factor=2.0):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.h_conv = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.v_conv = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.h_mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1))
        self.v_mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1))
        self.gate = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, feat):
        f_h = feat['l2r'] + feat['r2l']
        f_v = feat['t2b'] + feat['b2t']
        h_out = self.h_mlp(self.h_conv(f_h) + f_h + self.v_conv(f_v))
        v_out = self.v_mlp(self.v_conv(f_v) + f_v + self.h_conv(f_h))
        g = torch.sigmoid(self.gate)
        return g * h_out + (1 - g) * v_out


class DirectionMambaLayer(nn.Module):
    """Cross-Scan + S6 Block + CrossMerge + DSM."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.scan = CrossScan()
        self.merge = CrossMerge()
        self.s6 = SimpleMambaBlock(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dsm = DirectionSpecificMerge(dim=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        seqs, shape = self.scan(x)
        seqs_out = {}
        for k in seqs:
            s = self.norm(seqs[k])
            seqs_out[k] = self.s6(s) + seqs[k]
        directional = self.merge(seqs_out, shape)
        return self.dsm(directional) + residual


class DirectionMambaEncoder(nn.Module):
    """Stacked DirectionMambaLayer."""

    def __init__(self, dim, num_layers=2, d_state=16):
        super().__init__()
        self.layers = nn.ModuleList([DirectionMambaLayer(dim=dim, d_state=d_state) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)


class CrossScanBottleneckWrapper(nn.Module):
    """
    Wraps ViT model and inserts Cross-Scan Mamba bottleneck.
    Pipeline: ViT encoder -> proj 768->256 -> CrossScanMamba -> proj 256->768 -> decoder.
    """

    def __init__(self, vit_model, mamba_bottleneck, in_dim=768, mid_dim=256):
        super().__init__()
        self.vit = vit_model
        self.bottleneck = mamba_bottleneck
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
        ) if in_dim != mid_dim else nn.Identity()
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
        ) if mid_dim != in_dim else nn.Identity()

    def forward(self, x):
        B, _, H_in, W_in = x.shape
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x_vit, attn_weights, features = self.vit.transformer(x)
        
        # Handle both 1D tokens and 2D features
        if x_vit.dim() == 3:
            N = x_vit.size(1)
            H_patch = W_patch = int(N ** 0.5)
            D = x_vit.size(-1)
            x_2d = rearrange(x_vit, 'b (h w) d -> b d h w', h=H_patch, w=W_patch)
        else:
            x_2d = x_vit
        
        x_2d = self.in_proj(x_2d)          # 768 -> 256
        x_2d = self.bottleneck(x_2d)        # Cross-Scan Mamba
        x_2d = self.out_proj(x_2d)          # 256 -> 768
        
        if x_vit.dim() == 3:
            x_1d = rearrange(x_2d, 'b d h w -> b (h w) d')
        else:
            x_1d = x_2d
        
        x_dec = self.vit.decoder(x_1d, features)
        logits = self.vit.segmentation_head(x_dec)
        return nn.functional.softmax(logits, dim=1)
