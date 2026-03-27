import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from scale_with_openmeasure import scale_train_tensor
from models import Decoder, Encoder, ModelClass, Transformer


def build_windows(data, past_len, future_len):
    # data shape: (Nt, Nf, Nz, Nx)
    # returns shape: (N_samples, past_len + future_len, Nf, Nz, Nx)

    total = past_len + future_len
    n_samples = data.shape[0] - total + 1
    if n_samples <= 0:
        raise ValueError(
            f"Dataset too short (Nt={data.shape[0]}) for past_len={past_len}, future_len={future_len}"
        )

    windows = np.stack([data[i : i + total] for i in range(n_samples)], axis=0)
    return windows


def main():
    with open("train.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    print(f'reading file \"train.json\" ...')

    nt = settings["Nt"]
    nf = settings["Nf"]
    nz = settings["Nz"]
    nx = settings["Nx"]

    epochs = settings["epochs"]
    batch_size = settings["batch_size"]
    lr = settings["lr"]

    train_ratio = settings["train_ratio"]
    test_ratio = settings["test_ratio"]

    train_together = settings["train_together"]
    train_autoencoder = settings["train_autoencoder"]
    train_transformer = settings["train_transformer"]

    autoregressive = settings["autoregressive"]
    n_past = settings["n_past"]

    recursive = settings["recursive"]
    recursive_steps = settings["recursive_steps"]
    recursive_discount = settings["recursive_discount"]

    channels = settings["channels"]
    embed_dim = settings["transformer_embed_dim"]
    n_layers = settings["transformer_layers"]
    num_heads = settings["transformer_heads"]
    hidden_dim = settings["transformer_hidden_dim"]
    models_dir = settings["models_dir"]

    data_path = settings["data_path"]
    dataset = np.load(data_path)
    if dataset.shape != (nt, nf, nz, nx):
        raise ValueError(f"ERROR: Unexpected dataset shape {dataset.shape}, expected {(nt, nf, nz, nx)}")
    print(f'Dataset shape successfully interpreted: [{nt}, {nf}, {nz}, {nx}]')

    grid_path = settings["grid_path"]
    grid = np.load(grid_path)
    if grid.shape != (nz*nx,3):
        raise ValueError(f"ERROR: Unexpected grid shape {grid.shape}, expected {(nz*nx, 3)}")
    print(f'Grid shape successfully interpreted: [{nz} * {nx}, 3]')

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("ERROR: train_ratio must be between 0 and 1")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("ERROR: test_ratio must be between 0 and 1")
    if abs((train_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("ERROR: train_ratio + test_ratio must be 1")

    n_training = int(nt * train_ratio)
    n_test = nt - n_training

    if n_training <= 0 or n_test <= 0:
        raise ValueError(f"ERROR: Split non valido: n_training={n_training}, n_test={n_test}")

    train_snapshots = dataset[:n_training]
    test_snapshots = dataset[n_training:]
    print(f'\ntraining snapshots: {n_training}')
    print(f'testing snapshots : {n_test}')

    print(f'\nScaling training data ...')
    train_snapshots_scaled, rom = scale_train_tensor(train_snapshots, grid)
    print(f'Training data successfully scaled')

    past_len = n_past if autoregressive else 1
    print(f'\nAutoregressive training: {autoregressive}')
    if autoregressive: print(f'Nr. processed snapshots: {past_len}')

    future_len = recursive_steps if recursive else 1
    print(f'\nRecurrent training: {recursive}')
    if recursive: print(f'Nr. steps: {future_len}')

    windows = build_windows(train_snapshots_scaled, past_len=past_len, future_len=future_len)
    train_tensor = torch.tensor(windows, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)

    encoder = Encoder(nf, channels)
    decoder = Decoder(nf, list(reversed(channels)))

    latent_shape = (nz // (2 ** len(channels)), nx // (2 ** len(channels)))
    latent_channels = channels[-1]
    transformer_in_channels = latent_channels * (n_past if autoregressive else 1)

    transformer = Transformer(
        shape=latent_shape,
        input_channels=transformer_in_channels,
        output_channels=latent_channels,
        n_layers=n_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )

    model = ModelClass(encoder, decoder, transformer)
    model.set_datasets(train_dataset)

    print(f'\nCAE and Transformer trained together: {train_together}')

    if train_together == False:
        if train_autoencoder:
            model.set_optimizers(
                lr=lr,
                train_autoencoder=True,
                train_transformer=False,
                joint_training=False,
            )

            print(f'\nCAE training:')
            ae_time = model.train(
                epochs=epochs,
                batch_size=batch_size,
                autoregressive=autoregressive,
                n_past=n_past,
                recursive=recursive,
                recursive_steps=recursive_steps,
                recursive_discount=recursive_discount,
                checkpoint_dir=models_dir,
                save_checkpoint_each_epoch=False,
                checkpoint_label="model",
                stage_name="autoencoder",
                settings=settings,
            )
            print(f"Total CAE training time: {ae_time:.2f}s")

        if train_transformer:
            model.set_optimizers(
                lr=lr,
                train_autoencoder=False,
                train_transformer=True,
                joint_training=False,
            )

            print(f'\nTransformer training:')
            transformer_time = model.train(
                epochs=epochs,
                batch_size=batch_size,
                autoregressive=autoregressive,
                n_past=n_past,
                recursive=recursive,
                recursive_steps=recursive_steps,
                recursive_discount=recursive_discount,
                checkpoint_dir=models_dir,
                save_checkpoint_each_epoch=True,
                checkpoint_label="model",
                stage_name="transformer",
                settings=settings,
            )
            print(f"Total Transformer training time: {transformer_time:.2f}s")
    else:
        model.set_optimizers(
            lr=lr,
            train_autoencoder=train_autoencoder,
            train_transformer=train_transformer,
            joint_training=True,
        )

        print(f'\nCAE + Transformer training:')
        joint_time = model.train(
            epochs=epochs,
            batch_size=batch_size,
            autoregressive=autoregressive,
            n_past=n_past,
            recursive=recursive,
            recursive_steps=recursive_steps,
            recursive_discount=recursive_discount,
            checkpoint_dir=models_dir,
            save_checkpoint_each_epoch=True,
            checkpoint_label="model",
            stage_name="joint",
            settings=settings,
        )
        print(f"Total Joint training time: {joint_time:.2f}s")

    checkpoint_setting = settings.get("checkpoint_path", "trained_model.pt")
    final_checkpoint_path = Path(checkpoint_setting)
    if not final_checkpoint_path.is_absolute():
        final_checkpoint_path = Path(models_dir) / final_checkpoint_path
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "transformer": model.transformer.state_dict(),
            "settings": settings,
        },
        final_checkpoint_path,
    )

if __name__ == "__main__":
    main()
