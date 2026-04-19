from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaselineTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loaders: List[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: torch.device | str = "cuda",
        gradient_clip: float = 3.0,
        best_model_path: str | Path = "best_baseline.pt",
    ) -> None:
        self.model = model
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gradient_clip = gradient_clip
        self.best_model_path = Path(best_model_path)

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            x_t = batch["x_t"].to(self.device)
            x_seq = batch["x_seq"].to(self.device)

            batch_size = x_t.size(0)
            horizon = x_seq.shape[1]

            self.optimizer.zero_grad()
            pred_seq = self.model(x_t, horizon=horizon)
            loss = self.criterion(pred_seq, x_seq)

            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return {"training epoch loss": total_loss / max(total_samples, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, mode: str) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x_t = batch["x_t"].to(self.device)
            x_seq = batch["x_seq"].to(self.device)

            horizon = x_seq.shape[1]
            pred_seq = self.model(x_t, horizon=horizon)
            loss = self.criterion(pred_seq, x_seq)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {f"{mode} loss": avg_loss}

    def learn(self, num_epochs: int) -> Dict[str, List[float]]:
        best_val_loss = float("inf")
        train_losses: List[float] = []
        val_losses: List[float] = []

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
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_l)
                else:
                    self.scheduler.step()

            if v_l < best_val_loss:
                best_val_loss = v_l
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.best_model_path)

        return {"train epoch loss": train_losses, "val epoch loss": val_losses}

    def test(self) -> Dict[str, float]:
        if not self.best_model_path.exists():
            raise FileNotFoundError(
                f"best model path not found: {self.best_model_path}. Train first."
            )

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )

        test_loss = self.evaluate(self.test_loader, "test")
        tl = test_loss["test loss"]
        print(f"Test loss: {tl:.6f}")
        return test_loss
