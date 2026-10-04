"""
Swin Transformer Encoder (hierarchical, shifted-window self-attention).

This is a compact, self-contained implementation (no external timm dependency
required) that follows the standard Swin design: patch embedding -> 4 stages,
each with window-attention transformer blocks + patch merging (downsample x2
between stages). It outputs multi-scale feature maps consumed by the
Attention U-Net decoder for skip connections.
"""
import torch
import torch.nn as nn


def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    """One Swin transformer block with (optionally shifted) window attention + MLP."""

    def __init__(self, dim, num_heads, window_size=8, shift=False, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Downsamples resolution by 2x and doubles channel dim (like a strided conv)."""

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, patch_size=4, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        x = self.norm(x)
        return x, H, W


class SwinTransformerEncoder(nn.Module):
    """
    Hierarchical encoder producing 4 stages of multi-scale features, e.g. for a
    512x512 input with patch_size=4: stage resolutions 128,64,32,16 with
    channel dims 96,192,384,768 (default config).
    """

    def __init__(self, in_chans=1, patch_size=4, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=8, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, patch_size, embed_dim)
        self.num_stages = len(depths)

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            blocks = nn.ModuleList([
                SwinBlock(dim, num_heads[i], window_size, shift=(j % 2 == 1), dropout=dropout)
                for j in range(depth)
            ])
            self.stages.append(blocks)
            if i < self.num_stages - 1:
                self.merges.append(PatchMerging(dim))
                dim *= 2
        self.out_dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        features = []
        for i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, H, W)
            B, L, C = x.shape
            features.append(x.transpose(1, 2).view(B, C, H, W))
            if i < self.num_stages - 1:
                x, H, W = self.merges[i](x, H, W)
        return features  # list of 4 feature maps, coarsest last
