from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any,Dict
from ArtFire.DL.Models.MLP import CustomMLP


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encodings for sequence data. Implemented as a torch.nn layer.

    Positional encodings are added to input embeddings to provide information about the
    positions of tokens in the sequence. They are commonly used in transformer-based models.

    Attributes:
        encoding (torch.Tensor): A precomputed matrix of positional encodings with shape
                                 (1, max_len, d_model). The matrix is computed using
                                 sinusoidal functions.
    """


    def __init__(self, d_model: int , seq_len: int):
        """
        Initializes the PositionalEncoding class.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            seq_len (int): The length of the sequence.

        Behavior:
            - Precomputes a positional encoding matrix of shape (1, seq_len, d_model).
            - Uses sinusoidal functions to encode positions:
                - For even indices in the model dimension (2i):
                  PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
                - For odd indices (2i+1):
                  PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):

        """
        Adds positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input embeddings augmented with positional encodings.
                          The shape is the same as the input: (batch_size, seq_len, d_model).
       """
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_config: Dict[str, Any],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_config["input_dim"] = d_model
        mlp_config["output_dim"] = d_model

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp=CustomMLP(**mlp_config)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, d_model]
        # I prefered to apply normalization before the non-linear projections. The original transformer
        # paper applies after.
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class ARTransformerForecaster(nn.Module):
    """
    Autoregressive transformer forecaster.

    Input:
        z_t: [B, N, D_in] or [N, D_in]

    Output:
        preds: [B, horizon, N, D_in] or [horizon, N, D_in]

    The model learns a one-step transition:
        z_{t+1} = z_t + g(z_t)

    and rolls out autoregressively for the requested horizon.
    """

    def __init__(
        self,
        n_tokens: int = 63,
        token_dim: int = 2880,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        mlp_config: Dict[str,Any] =  {"hidden_layers":[128], "dropout_p": 0.0, "normalization":"Batch-Norm",
                 "activation":"leaky_relu", "seed":42, "initialization": "kaiming uniform"},
        dropout: float = 0.0,
        pos_encoding : str= "sinusoidal",  # either sinusoidal or learnable
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:  # each head takes x as input and returns a vetor of shape d_model/n_heads, then all
                                    # outputs are conatenated
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.d_model = d_model
        self.use_residual = use_residual

        self.input_proj = nn.Linear(token_dim, d_model)

        self.pos_embed=PositionalEncoding(d_model=d_model, seq_len=n_tokens) if pos_encoding=="sinusoidal" else LearnablePositionalEncoding(n_tokens=n_tokens, d_model=d_model)


        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_config=mlp_config,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, token_dim)

    def _forward_one_step(self, z: Tensor) -> Tensor:
        """
        One-step prediction.

        Args:
            z: [B, N, D_in]

        Returns:
            z_next: [B, N, D_in]
        """
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape [B, N, D], got {tuple(z.shape)}")

        bsz, n_tokens, token_dim = z.shape
        if n_tokens != self.n_tokens:
            raise ValueError(
                f"Expected {self.n_tokens} tokens, got {n_tokens}."
            )
        if token_dim != self.token_dim:
            raise ValueError(
                f"Expected token dim {self.token_dim}, got {token_dim}."
            )

        x = self.input_proj(z)                  # [B, N, d_model]
        x =  self.pos_embed(x)                 # token identity encoding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        delta = self.output_proj(x)             # [B, N, D_in]

        if self.use_residual:
            z_next = z + delta
        else:
            z_next = delta

        return z_next

    def forward(self, z_t: Tensor, horizon: int) -> Tensor:
        """
        Autoregressive rollout.

        Args:
            z_t: [B, N, D_in] or [N, D_in]
            horizon: number of future steps to predict

        Returns:
            preds:
                [B, horizon, N, D_in] if input was batched
                [horizon, N, D_in] if input was unbatched
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        squeeze_batch = False
        if z_t.ndim == 2:  # if it is a single element to predict in a batch (no batch index)
            z_t = z_t.unsqueeze(0)   # [1, N, D]
            squeeze_batch = True

        if z_t.ndim != 3:
            raise ValueError(
                f"Expected z_t with shape [N, D] or [B, N, D], got {tuple(z_t.shape)}"
            )

        preds = []
        z = z_t

        for _ in range(horizon):
            z = self._forward_one_step(z)   # autoregressive update
            preds.append(z)

        preds = torch.stack(preds, dim=1)   # [B, H, N, D]

        if squeeze_batch:
            preds = preds.squeeze(0)        # [H, N, D]

        return preds