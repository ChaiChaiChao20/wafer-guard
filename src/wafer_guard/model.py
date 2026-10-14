from __future__ import annotations

import torch
from torch import nn


class WaferGuardSAE(nn.Module):
    """Compact supervised autoencoder with an explicitly aligned batch axis."""

    def __init__(self, input_features: int, hidden: int = 128, latent: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_features),
        )
        self.classifier = nn.Linear(latent, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, features], received {tuple(x.shape)}")
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        logits = self.classifier(latent).squeeze(-1)
        return reconstruction, logits

