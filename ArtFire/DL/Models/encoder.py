import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """
    Transformer encoder module.

    Expected input shape:
        - if batch_first=True:  (batch_size, seq_len, input_dim)
        - if batch_first=False: (seq_len, batch_size, input_dim)

    This module optionally projects input_dim -> d_model, adds positional embeddings,
    and applies a stack of Transformer encoder blocks.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        max_seq_len: int = 4096,
        use_input_projection: bool = True,
        use_positional_embedding: bool = True,
        final_norm: bool = True,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )
        if input_dim <= 0 or d_model <= 0 or nhead <= 0 or num_layers <= 0:
            raise ValueError("input_dim, d_model, nhead, and num_layers must be > 0.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0.")

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.use_positional_embedding = use_positional_embedding

        # Optional input projection
        if use_input_projection or input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model, bias=bias)
        else:
            if input_dim != d_model:
                raise ValueError(
                    "If use_input_projection=False, then input_dim must equal d_model."
                )
            self.input_projection = nn.Identity()

        # Learned positional embeddings
        if use_positional_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        else:
            self.register_parameter("pos_embedding", None)

        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
        )

        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if final_norm else None

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Reinitialize projection
        if isinstance(self.input_projection, nn.Linear):
            nn.init.xavier_uniform_(self.input_projection.weight)
            if self.input_projection.bias is not None:
                nn.init.zeros_(self.input_projection.bias)

        # Reinitialize positional embeddings
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # Reinitialize each encoder layer explicitly
        for layer in self.encoder.layers:
            self._reset_transformer_layer(layer)

        # Final norm
        if self.encoder.norm is not None:
            nn.init.ones_(self.encoder.norm.weight)
            nn.init.zeros_(self.encoder.norm.bias)

    @staticmethod
    def _reset_transformer_layer(layer: nn.TransformerEncoderLayer) -> None:
        # Multi-head attention
        if hasattr(layer.self_attn, "in_proj_weight") and layer.self_attn.in_proj_weight is not None:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
        if hasattr(layer.self_attn, "in_proj_bias") and layer.self_attn.in_proj_bias is not None:
            nn.init.zeros_(layer.self_attn.in_proj_bias)
        if hasattr(layer.self_attn, "out_proj"):
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            if layer.self_attn.out_proj.bias is not None:
                nn.init.zeros_(layer.self_attn.out_proj.bias)

        # Feedforward
        nn.init.xavier_uniform_(layer.linear1.weight)
        if layer.linear1.bias is not None:
            nn.init.zeros_(layer.linear1.bias)

        nn.init.xavier_uniform_(layer.linear2.weight)
        if layer.linear2.bias is not None:
            nn.init.zeros_(layer.linear2.bias)

        # Layer norms
        nn.init.ones_(layer.norm1.weight)
        nn.init.zeros_(layer.norm1.bias)
        nn.init.ones_(layer.norm2.weight)
        nn.init.zeros_(layer.norm2.bias)

    def forward(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Args:
            x:
                Input tensor of shape
                - (B, S, input_dim) if batch_first=True
                - (S, B, input_dim) if batch_first=False
            src_mask:
                Optional attention mask of shape (S, S) or broadcast-compatible.
            src_key_padding_mask:
                Optional padding mask of shape
                - (B, S) if batch_first=True
                - (B, S) also for batch_first=False, as required by PyTorch.
                True means "ignore this token".
            is_causal:
                Whether to apply causal masking semantics.

        Returns:
            Encoded tensor of shape
            - (B, S, d_model) if batch_first=True
            - (S, B, d_model) if batch_first=False
        """
        x = self.input_projection(x)

        if self.use_positional_embedding:
            if self.batch_first:
                seq_len = x.size(1)
                if seq_len > self.pos_embedding.size(1):
                    raise ValueError(
                        f"Sequence length ({seq_len}) exceeds max_seq_len "
                        f"({self.pos_embedding.size(1)})."
                    )
                x = x + self.pos_embedding[:, :seq_len, :]
            else:
                seq_len = x.size(0)
                if seq_len > self.pos_embedding.size(1):
                    raise ValueError(
                        f"Sequence length ({seq_len}) exceeds max_seq_len "
                        f"({self.pos_embedding.size(1)})."
                    )
                x = x + self.pos_embedding[:, :seq_len, :].transpose(0, 1)

        x = self.input_dropout(x)

        x = self.encoder(
            src=x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        return x