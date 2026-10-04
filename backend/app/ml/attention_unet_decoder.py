"""
Attention U-Net Decoder.

Consumes the multi-scale feature pyramid from the Swin encoder and upsamples
progressively back to full resolution, using Attention Gates on each skip
connection (Oktay et al., "Attention U-Net") so the decoder learns to focus
on salient anatomical regions before fusing encoder features.

Two output heads:
    - heatmap head : NUM_LANDMARKS channels, one Gaussian-heatmap per landmark
    - offset head  : 2 * NUM_LANDMARKS channels, sub-pixel (dx, dy) regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Gating signal (from coarser decoder level) attends over the skip
    connection (from encoder) before concatenation."""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.W_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.relu(g + s)
        psi = self.psi(psi)
        return skip * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn_gate = AttentionGate(out_ch, skip_ch, out_ch // 2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionUNetDecoder(nn.Module):
    def __init__(self, encoder_dims, num_landmarks=None, dropout=None):
        """encoder_dims: e.g. [96, 192, 384, 768] (fine -> coarse) from the Swin encoder."""
        super().__init__()
        num_landmarks = num_landmarks or settings.NUM_LANDMARKS
        dropout = dropout if dropout is not None else settings.DROPOUT_RATE

        c1, c2, c3, c4 = encoder_dims  # fine->coarse

        self.up3 = UpBlock(c4, c3, c3, dropout)
        self.up2 = UpBlock(c3, c2, c2, dropout)
        self.up1 = UpBlock(c2, c1, c1, dropout)

        # Two final upsamples to recover the 4x patch-embedding downsample back to full res
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(c1 // 2, c1 // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Conv2d(c1 // 2, num_landmarks, kernel_size=1)
        self.offset_head = nn.Conv2d(c1 // 2, num_landmarks * 2, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4 = features  # fine -> coarse
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.final_up(x)

        heatmaps = self.heatmap_head(x)          # (B, num_landmarks, H, W)
        offsets = self.offset_head(x)             # (B, num_landmarks*2, H, W)  -> (dx, dy) per landmark
        return heatmaps, offsets
