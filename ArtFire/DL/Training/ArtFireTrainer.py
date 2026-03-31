from ArtFire.DL.Models.Forecast import ARTransformerForecaster
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from ArtFire.DL.Loss.Loss import Loss
from ArtFire.DL.Loss.ArtFireLoss import (
    MAEReconstructionLoss,
    MSEReconstructionLoss,
    SmoothL1ReconstructionLoss,
)


class ArtFireTrainer:
    def __init__(
        self,
        model: ARTransformerForecaster,
        loaders: List[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: Loss = MSEReconstructionLoss(),
        scheduler: torch.optim.lr_scheduler = None,
        device: torch.device = "cuda",
        gradient_clip=3,
    ):
        self.model = model
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gradient_clip = gradient_clip

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            x_t = batch["x_t"].to(self.device)
            x_seq = batch["x_seq"].to(self.device)

            B = x_t.size(0)
            horizon = x_seq.shape[1]

            self.optimizer.zero_grad()
            pred_seq = self.model(x_t, horizon=horizon)

            _, loss = self.criterion(pred_seq, x_seq)

            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

        return {"training epoch loss": total_loss / total_samples}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, mode: str) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        time_loss = None

        for batch in loader:
            x_t = batch["x_t"].to(self.device)  # [B, N, D]
            x_seq = batch["x_seq"].to(self.device)  # [B, H, N, D]

            horizon = x_seq.shape[1]
            if time_loss is None:
                time_loss = torch.zeros(
                    horizon, requires_grad=False, device=self.device
                )

            pred_seq = self.model(x_t, horizon=horizon)
            loss, loss_m = self.criterion(pred_seq, x_seq)

            time_loss += loss

            total_loss += loss_m.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        time_loss = time_loss / max(n_batches, 1)
        return {f"{mode} loss": avg_loss, f"{mode} time loss": time_loss}

    def learn(self, num_epochs: int):
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        for epoch in tqdm(range(num_epochs)):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate(self.val_loader, "val epoch")
            t_l = train_loss["training epoch loss"]
            v_l = val_loss["val epoch loss"]
            train_losses.append(t_l)
            val_losses.append(v_l)

            print(
                f"\nEpoch {epoch + 1:03d} | train loss: {t_l:.6f} | val loss: {v_l:.6f}"
            )
            if self.scheduler is not None:
                # for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(v_l)
                else:
                    self.scheduler.step()
            if v_l < best_val_loss:
                best_val_loss = v_l
                torch.save(self.model.state_dict(), "best_artfire.pt")

    def test(self):
        best_model_path = Path("best_artfire.pt")
        if not best_model_path.exists():
            raise FileNotFoundError(f"best_model_path not found, first train the model")
        self.model.load_state_dict(
            torch.load("best_artfire.pt", map_location=self.device)
        )
        test_loss = self.evaluate(self.test_loader, "test")
        tl = test_loss["test loss"]
        print(f"Test loss: {tl:.6f}")
        return test_loss
