from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn


class ConvAutoEncoder(nn.Module):
    """CAE for fields shaped [B, Nf, Nz, Nx] with 4 conv stages.

    Encoder channels: Nf -> 32 -> 64 -> 128 -> 256
    """

    def __init__(
        self,
        in_channels: int = 13,
        channel_progression: Sequence[int] = (32, 64, 128, 256),
        latent_dim: int = 256,
        input_hw: Tuple[int, int] = (256, 64),
    ) -> None:
        super().__init__()

        channels = [in_channels, *channel_progression]
        encoder_layers: List[nn.Module] = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            encoder_layers.extend(
                [
                    nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(cout),
                    nn.GELU(),
                ]
            )
        self.encoder_cnn = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            encoded = self.encoder_cnn(dummy)
            self.encoded_shape = tuple(encoded.shape[1:])
            flat_dim = int(encoded.numel())

        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        decoder_layers: List[nn.Module] = []
        rev_channels = list(channels[::-1])
        for idx, (cin, cout) in enumerate(zip(rev_channels[:-1], rev_channels[1:])):
            decoder_layers.append(
                nn.ConvTranspose2d(cin, cout, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            if idx < len(rev_channels) - 2:
                decoder_layers.extend([nn.BatchNorm2d(cout), nn.GELU()])
        self.decoder_cnn = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        h = torch.flatten(h, start_dim=1)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(z.shape[0], *self.encoded_shape)
        return self.decoder_cnn(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
