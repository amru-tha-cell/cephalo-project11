"""
Full model: Swin Transformer Encoder -> Attention U-Net Decoder ->
19 Landmark Heatmaps + 2D Offset Regression.

Dropout layers are embedded throughout the decoder (see attention_unet_decoder.py)
so Monte Carlo Dropout can be enabled at inference time (see mc_dropout.py)
without any architecture change.
"""
import torch
import torch.nn as nn

from app.core.config import settings
from app.ml.swin_encoder import SwinTransformerEncoder
from app.ml.attention_unet_decoder import AttentionUNetDecoder


class SwinAttentionUNet(nn.Module):
    def __init__(self, num_landmarks=None, in_chans=1, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=8,
                 dropout=None):
        super().__init__()
        num_landmarks = num_landmarks or settings.NUM_LANDMARKS
        dropout = dropout if dropout is not None else settings.DROPOUT_RATE

        self.encoder = SwinTransformerEncoder(
            in_chans=in_chans, embed_dim=embed_dim, depths=depths,
            num_heads=num_heads, window_size=window_size, dropout=dropout,
        )
        self.decoder = AttentionUNetDecoder(
            encoder_dims=self.encoder.out_dims, num_landmarks=num_landmarks, dropout=dropout,
        )

    def forward(self, x):
        """x: (B, 1, H, W) preprocessed grayscale image, H=W=settings.IMAGE_SIZE.
        Returns heatmaps (B, N, H, W) and offsets (B, 2N, H, W)."""
        features = self.encoder(x)
        heatmaps, offsets = self.decoder(features)
        return heatmaps, offsets


def build_model(weights_path: str = None, device: str = "cpu") -> SwinAttentionUNet:
    """Instantiate the model and load trained weights if available.
    If no checkpoint exists yet (e.g. before training on the ISBI dataset),
    returns a randomly-initialized model so the API stays fully runnable end-to-end."""
    model = SwinAttentionUNet()
    weights_path = weights_path or settings.WEIGHTS_PATH
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[model] Loaded trained weights from {weights_path}")
    except (FileNotFoundError, RuntimeError, EOFError):
        print(f"[model] No valid checkpoint at {weights_path} -- using randomly-initialized "
              f"weights (train on the ISBI dataset and save to this path).")
    model.to(device)
    return model
