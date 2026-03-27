import numpy as np
from pathlib import Path
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(
        self,
        n_features,
        channels,
        pool_mode="avg",
        activation=nn.LeakyReLU,
        padding_mode="circular",
    ):
        super().__init__()

        cs = [n_features] + channels
        layers = []

        for i in range(len(cs) - 1):
            if pool_mode == "stride":
                layers.append(
                    nn.Conv2d(
                        cs[i],
                        cs[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        padding_mode=padding_mode,
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        cs[i],
                        cs[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode=padding_mode,
                    )
                )
                if pool_mode == "avg":
                    layers.append(nn.AvgPool2d(2))
                elif pool_mode == "max":
                    layers.append(nn.MaxPool2d(2))

            layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        n_features,
        channels,
        up_mode="bilinear",
        activation=nn.LeakyReLU,
        padding_mode="circular",
    ):
        super().__init__()

        # channels must be passed from latent -> high resolution, e.g. [256, 128, 64, 32]
        cs = channels + [n_features]
        layers = []

        for i in range(len(cs) - 1):
            is_last = i == len(cs) - 2
            act = nn.Identity if is_last else activation

            if up_mode == "stride":
                layers.append(
                    nn.ConvTranspose2d(cs[i], cs[i + 1], kernel_size=4, stride=2, padding=1)
                )
            else:
                layers.append(nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False))
                layers.append(
                    nn.Conv2d(
                        cs[i],
                        cs[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode=padding_mode,
                    )
                )

            layers.append(act())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(
        self,
        shape,
        input_channels,
        output_channels,
        n_layers=4,
        embed_dim=128,
        num_heads=8,
        hidden_dim=256,
        periodic=True,
    ):
        super().__init__()

        self.spatial_shape = shape
        nz, nx = shape

        if periodic:
            z, x = torch.meshgrid(torch.arange(nz), torch.arange(nx), indexing="ij")
            z_freq = torch.fft.rfftfreq(nz)[1:, None, None]
            x_freq = torch.fft.rfftfreq(nx)[1:, None, None]

            z_sin = torch.sin(2 * np.pi * z_freq * z)
            z_cos = torch.cos(2 * np.pi * z_freq * z)
            x_sin = torch.sin(2 * np.pi * x_freq * x)
            x_cos = torch.cos(2 * np.pi * x_freq * x)

            pos_info = torch.cat([z_sin, z_cos, x_sin, x_cos])
        else:
            z, x = torch.meshgrid(
                torch.arange(1, nz + 1) / nz,
                torch.arange(1, nx + 1) / nx,
                indexing="ij",
            )
            pos_info = torch.stack([z, x])

        dim_pos = pos_info.shape[0]
        self.register_buffer("pos_info", pos_info.unsqueeze(0), persistent=False)

        self.in_proj = nn.Conv2d(input_channels, embed_dim, kernel_size=1)
        self.pos_embedder = nn.Sequential(
            nn.Conv2d(dim_pos, dim_pos * 4, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim_pos * 4, embed_dim, 1),
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mha": nn.MultiheadAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            batch_first=True,
                        ),
                        "ff": nn.Sequential(
                            nn.Conv1d(embed_dim, hidden_dim, 1),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dim, embed_dim, 1),
                        ),
                        "norm1": nn.LayerNorm(embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        self.out_proj = nn.Conv2d(embed_dim, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_proj(x)
        x = x + self.pos_embedder(self.pos_info)

        bsz, ch, nz, nx = x.shape
        x = x.view(bsz, ch, -1).permute(0, 2, 1)  # (B, N, C)

        for layer in self.layers:
            attn_out, _ = layer["mha"](x, x, x, need_weights=False)
            x = layer["norm1"](x + attn_out)

            x_ff = layer["ff"](x.permute(0, 2, 1)).permute(0, 2, 1)
            x = layer["norm2"](x + x_ff)

        x = x.permute(0, 2, 1).view(bsz, ch, nz, nx)
        return self.out_proj(x)


class ModelClass:
    def __init__(self, encoder, decoder, transformer, device=Device):
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.transformer = transformer.to(device)

    def set_datasets(self, train_dataset, val_dataset=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.has_val = val_dataset is not None

    def set_optimizers(
        self,
        lr=1e-3,
        train_autoencoder=True,
        train_transformer=True,
        joint_training=True,
    ):
        self.train_autoencoder = train_autoencoder
        self.train_transformer = train_transformer
        self.joint_training = joint_training

        if joint_training:
            params = []
            if train_autoencoder:
                params += list(self.encoder.parameters()) + list(self.decoder.parameters())
            if train_transformer:
                params += list(self.transformer.parameters())
            self.optimizer = optim.Adam(params, lr=lr)
        else:
            if train_autoencoder:
                self.optimizer_ae = optim.Adam(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
                )
            if train_transformer:
                self.optimizer_t = optim.Adam(self.transformer.parameters(), lr=lr)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def _predict_next_latent(self, latent_history, n_past):
        if n_past > 1:
            z_in = torch.cat(latent_history[-n_past:], dim=1)
        else:
            z_in = latent_history[-1]
        return self.transformer(z_in)

    def train(
        self,
        epochs=50,
        batch_size=32,
        criterion=nn.MSELoss(),
        autoregressive=False,
        n_past=1,
        recursive=False,
        recursive_steps=1,
        recursive_discount=0.8,
        checkpoint_dir=None,
        save_checkpoint_each_epoch=False,
        checkpoint_label="model",
        stage_name=None,
        settings=None,
    ):
        loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        stage_label = stage_name if stage_name is not None else "training"
        training_start_time = time.perf_counter()
        for epoch in range(epochs):
            epoch_start_time = time.perf_counter()
            epoch_losses = []

            for (x_seq,) in loader:
                x_seq = x_seq.to(self.device)  # (B, T, C, H, W)
                if not torch.isfinite(x_seq).all():
                    raise ValueError(
                        "Training data contains NaN or Inf values. "
                        "Please clean/normalize the dataset before training."
                    )

                if self.joint_training:
                    self.optimizer.zero_grad()
                else:
                    if self.train_autoencoder:
                        self.optimizer_ae.zero_grad()
                    if self.train_transformer:
                        self.optimizer_t.zero_grad()

                total_loss = 0.0

                if self.train_autoencoder:
                    x_rec = self.decode(self.encode(x_seq[:, 0]))
                    loss_ae = criterion(x_rec, x_seq[:, 0])
                    total_loss = total_loss + loss_ae

                if self.train_transformer:
                    past_len = n_past if autoregressive else 1
                    if x_seq.shape[1] < past_len + 1:
                        raise ValueError(
                            f"Window too short: got T={x_seq.shape[1]}, required at least {past_len + 1}"
                        )

                    latent_history = [self.encode(x_seq[:, i]) for i in range(past_len)]

                    steps = recursive_steps if recursive else 1
                    loss_dyn = 0.0

                    for step in range(steps):
                        z_pred = self._predict_next_latent(latent_history, past_len)
                        z_true = self.encode(x_seq[:, past_len + step])

                        weight = recursive_discount**step if recursive else 1.0
                        loss_dyn = loss_dyn + weight * criterion(z_pred, z_true)

                        latent_history.append(z_pred)

                    total_loss = total_loss + loss_dyn

                if not torch.isfinite(total_loss):
                    raise RuntimeError(
                        "Encountered non-finite loss (NaN/Inf) during training. "
                        "Aborting before writing invalid checkpoints."
                    )

                total_loss.backward()

                if self.joint_training:
                    self.optimizer.step()
                else:
                    if self.train_autoencoder:
                        self.optimizer_ae.step()
                    if self.train_transformer:
                        self.optimizer_t.step()

                epoch_losses.append(float(total_loss.item()))

            epoch_loss = float(np.mean(epoch_losses))
            epoch_duration = time.perf_counter() - epoch_start_time
            print(
                f"[{stage_label}] Epoch {epoch + 1}/{epochs} - "
                f"loss: {epoch_loss:.6f} - epoch_time: {epoch_duration:.2f}s"
            )
            if not math.isfinite(epoch_loss):
                raise RuntimeError(
                    f"Epoch {epoch + 1} produced non-finite loss ({epoch_loss}). "
                    "Training stopped to avoid saving corrupted checkpoints."
                )

            if save_checkpoint_each_epoch and checkpoint_path is not None:
                formatted_epoch = f"{epoch + 1:03d}"
                formatted_loss = f"{epoch_loss:.6f}"
                checkpoint_file = checkpoint_path / (
                    f"{checkpoint_label}_{formatted_epoch}_{formatted_loss}.pt"
                )
                torch.save(
                    {
                        "encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "transformer": self.transformer.state_dict(),
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "stage": stage_name,
                        "settings": settings,
                    },
                    checkpoint_file,
                )

        total_training_duration = time.perf_counter() - training_start_time
        return total_training_duration

    def predict_next(self, x_context, autoregressive=False, n_past=1):
        self.encoder.eval()
        self.decoder.eval()
        self.transformer.eval()

        with torch.no_grad():
            if x_context.dim() == 3:
                x_context = x_context.unsqueeze(0).unsqueeze(0)  # (1,1,C,H,W)
            elif x_context.dim() == 4:
                x_context = x_context.unsqueeze(0)  # (1,T,C,H,W)

            x_context = x_context.to(self.device)
            past_len = n_past if autoregressive else 1

            latent_history = [self.encode(x_context[:, i]) for i in range(past_len)]
            z_next = self._predict_next_latent(latent_history, past_len)
            x_next = self.decode(z_next)

        return x_next.squeeze(0)
