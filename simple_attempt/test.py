import json
import re
import math
from pathlib import Path
import numpy as np
import torch
import time
from scale_with_openmeasure import scale_train_tensor, scale_test_tensor, rescale_back_output
from models import Decoder, Encoder, ModelClass, Transformer
import os
import matplotlib.pyplot as plt


def _find_best_checkpoint(models_dir, start_epoch, output_dir):

    models_path = Path(models_dir)
    print(f"\nOpening folder {models_path}...")
    if not models_path.exists():
        raise FileNotFoundError(f"ERROR: models_dir does not exist: {models_path}")

    print("\nReading models...")

    pattern = re.compile(r"model_(\d+)_(\d+\.\d+)\.pt")

    epochs = []
    losses = []

    for checkpoint_file in models_path.glob("model_*_*.pt"):
        match = pattern.match(checkpoint_file.name)
        if not match:
            continue

        try:
            epoch = int(match.group(1))
            loss = float(match.group(2))
        except ValueError:
            continue

        if not math.isfinite(loss):
            continue

        epochs.append(epoch)
        losses.append(loss)

    if len(epochs) == 0:
        raise FileNotFoundError(
            f"ERROR: No checkpoint matching 'model_{{epoch}}_{{loss}}.pt' found in {models_path}"
        )

    # ordina per epoca
    data = sorted(zip(epochs, losses), key=lambda x: x[0])
    epochs, losses = zip(*data)
    epochs = list(epochs)
    losses = list(losses)

    print(f"- training epochs: {len(epochs)}")

    # best checkpoint
    min_loss = min(losses)
    best_idx = losses.index(min_loss)
    best_epoch = epochs[best_idx]
    best_file = models_path / f"model_{best_epoch:03d}_{min_loss}.pt"

    print(f"- best epoch: {best_epoch}")
    print(f"- min loss: {min_loss}")

    # filtro per plot
    filtered_epochs = [e for e in epochs if e >= start_epoch]
    filtered_losses = [l for e, l in zip(epochs, losses) if e >= start_epoch]

    plt.figure()
    plt.plot(filtered_epochs, filtered_losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss trend")

    output_path = output_dir / "loss_trend.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("- loss trend plot saved")

    return best_file, min_loss


def main():
    with open("test.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    print(f'\nReading file \"test.json\" ...')

    nt = settings["Nt"]
    nf = settings["Nf"]
    nz = settings["Nz"]
    nx = settings["Nx"]

    train_ratio = settings["train_ratio"]
    test_ratio = settings["test_ratio"]

    autoregressive = settings["autoregressive"]
    n_past = settings["n_past"]
    n_teststeps = settings["N_teststeps"]

    channels = settings["channels"]
    embed_dim = settings["transformer_embed_dim"]
    n_layers = settings["transformer_layers"]
    num_heads = settings["transformer_heads"]
    hidden_dim = settings["transformer_hidden_dim"]

    models_dir = settings["models_dir"]
    test_dir = settings["test_dir"]

    plot_start_epoch = settings["plot_start_epoch"]
    features = settings["features"]

    data_path = settings["data_path"]
    dataset = np.load(data_path)
    if dataset.shape != (nt, nf, nz, nx):
        raise ValueError(f"ERROR: Unexpected dataset shape {dataset.shape}, expected {(nt, nf, nz, nx)}")
    print(f'\nDataset shape successfully interpreted: [{nt}, {nf}, {nz}, {nx}]')

    grid_path = settings["grid_path"]
    grid = np.load(grid_path)
    if grid.shape != (nz*nx,3):
        raise ValueError(f"ERROR: Unexpected grid shape {grid.shape}, expected {(nz*nx, 3)}")
    print(f'\nGrid shape successfully interpreted: [{nz*nx}, 3]')

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("ERROR: train_ratio must be between 0 and 1")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("ERROR: test_ratio must be between 0 and 1")
    if abs((train_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("ERROR: train_ratio + test_ratio must be 1")
    if n_teststeps <= 0:
        raise ValueError("ERROR: N_teststeps must be >= 1")
    if autoregressive and n_past < 1:
        raise ValueError("ERROR: n_past must be >= 1 when autoregressive is true")

    n_training = int(nt * train_ratio)
    n_test = nt - n_training
    if n_training <= 0 or n_test <= 0:
        raise ValueError(f"ERROR: Split non valido: n_training={n_training}, n_test={n_test}")

    train_snapshots = dataset[:n_training]
    test_snapshots = dataset[n_training:]
    print(f'\ntraining snapshots: {n_training}')
    print(f'testing snapshots : {n_test}')

    print(f'\nScaling test data ...')
    train_snapshots_scaled, rom = scale_train_tensor(train_snapshots, grid)
    test_snapshots_scaled = scale_test_tensor(test_snapshots, rom)
    print(f'Testing data successfully scaled')

    required_seed = n_past if autoregressive else 1
    if test_snapshots.shape[0] < required_seed:
        raise ValueError(
            f"ERROR: Test set too short for seed. Got {test_snapshots.shape[0]} snapshots, "
            f"required at least {required_seed}"
        )
    print(f'\nAutoregressive training: {autoregressive}')
    if autoregressive: print(f'Nr. processed snapshots: {past_len}')

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

    folder_name = f"test_t{n_teststeps}_f{nf}_z{nz}_x{nx}"
    output_dir = Path(test_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path, checkpoint_loss = _find_best_checkpoint(models_dir, plot_start_epoch, output_dir)
    print(f"\nUsing best checkpoint: {checkpoint_path.name} (loss={checkpoint_loss:.6f})")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.transformer.load_state_dict(checkpoint["transformer"])

    print(f'\nStart testing ...')
    testing_start_time = time.time()

    generated = []
    if autoregressive:
        history = [test_snapshots[i] for i in range(n_past)]
        for _ in range(n_teststeps):
            x_context = np.stack(history[-n_past:], axis=0)
            x_context_tensor = torch.tensor(x_context, dtype=torch.float32)
            x_next = model.predict_next(x_context_tensor, autoregressive=True, n_past=n_past)
            x_next_np = x_next.detach().cpu().numpy()
            history.append(x_next_np)
            generated.append(x_next_np)
    else:
        current = torch.tensor(test_snapshots[0], dtype=torch.float32)
        for _ in range(n_teststeps):
            x_next = model.predict_next(current, autoregressive=False, n_past=1)
            x_next_np = x_next.detach().cpu().numpy()
            generated.append(x_next_np)
            current = torch.tensor(x_next_np, dtype=torch.float32)
    
    testing_end_time = time.time()
    print(f'End test!')
    print(f'-> elapsed time: {testing_end_time-testing_start_time:.2} s')

    generated_array_scaled = np.stack(generated, axis=0).astype(np.float32)
    generated_array = rescale_back_output(generated_array_scaled, rom)
    print(f'\n----> generated array shape: {generated_array.shape}')

    output_file = output_dir / f"T_t{n_teststeps}_f{nf}_z{nz}_x{nx}.npy"
    np.save(output_file, generated_array)

    print(f"\nSaved generated test rollout to: {output_file}")

    # start plot:
    num_plots = 5
    ts = np.linspace(0, n_teststeps-1, num_plots, dtype=int)
    for f in range(nf):

        plot_dir = os.path.join(output_dir, f"{features[f]}")
        os.makedirs(plot_dir, exist_ok=True)

        rmse = np.zeros(n_teststeps)
        diff = np.zeros([n_teststeps, nz, nx])
        err = np.zeros(n_teststeps)
        for t in range(n_teststeps):
            diff[t,:,:] = test_snapshots[t+1, f, :, :] - generated_array[t, f, :, :]
            err[t] = np.linalg.norm(diff[t,:,:]/test_snapshots[t, f, :, :])
            rmse[t] = np.sqrt(np.mean(diff[t,:,:]**2))
        
        plt.figure(figsize=(8,4))
        plt.plot(np.arange(n_teststeps), rmse, '-o', markersize=3)
        plt.xlabel("Timestep")
        plt.ylabel("RMSE")
        plt.title(f"RMSE evolution - feature {features[f]}")
        # plt.ylim(0,100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "RMSE_evolution.png"))
        plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(np.arange(n_teststeps), 100*err, '-o', markersize=3)
        plt.xlabel("Timestep")
        plt.ylabel("error (%)")
        plt.title(f"% error - feature {features[f]}")
        # plt.ylim(0,100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Err_evolution.png"))
        plt.close()

        fig, axs = plt.subplots(3, num_plots, figsize=(3*num_plots, 4*3))
        for i, t in enumerate(ts):

            # CFD
            #cmap='CMRmap'
            vmin = np.min([test_snapshots[t+1, f, :, :]]) #,generated_array[t, f, :, :]])
            vmax = np.max([test_snapshots[t+1, f, :, :]]) #,generated_array[t, f, :, :]])
            axs[0, i].pcolormesh(test_snapshots[t+1, f, :, :], cmap='CMRmap', vmin=vmin, vmax=vmax)
            axs[0, i].set_title(f"CFD - t={n_training+t+1}")
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])

            # ROMER
            axs[1, i].pcolormesh(generated_array[t, f, :, :], cmap='CMRmap', vmin=vmin, vmax=vmax)
            axs[1, i].set_title("ArtFire")
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])

            # DIFFERENZA %
            perc = 100 * diff[t,:,:]/test_snapshots[t+1, f, :, :]
            axs[2, i].pcolormesh(perc, cmap='bwr', vmin=-100, vmax=100)
            axs[2, i].set_title("Diff. %")
            axs[2, i].set_xticks([])
            axs[2, i].set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "comparison_grid.png"))
        plt.close()

    print("\nAll plots saved succesfully!")
        

    print(f"\nTest completed succesfully!\n\n")


if __name__ == "__main__":
    main()
