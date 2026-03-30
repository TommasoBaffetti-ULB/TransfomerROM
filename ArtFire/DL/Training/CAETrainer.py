from ArtFire.DL.Models.CAE import CAE
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import List


class CAETrainer:
    def __init__(self,model: CAE,
                 loaders: List[DataLoader],
                 optimizer: torch.optim.Optimizer,
                 criterion:torch.nn=torch.nn.MSELoss(),
                 scheduler: torch.optim.lr_scheduler=None,
                 device:torch.device= "cuda",
                 gradient_clip:float=3):
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
        running_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            B = x.size(0)

            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            if self.gradient_clip>0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                            self.gradient_clip)
            self.optimizer.step()

            running_loss += loss.item() * B
            total_samples += B

        return {"training epoch loss": running_loss / total_samples}

    @torch.no_grad()
    def evaluate(self,loader, mode:str):
        self.model.eval()
        running_loss = 0.0
        total_samples=0

        for batch in loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

            running_loss += loss.item() * x.size(0)  # loss is averaged over the batch dimension, this allows for dataset loss
            total_samples+=x.size(0)
        return {f"{mode} loss": running_loss / total_samples}

    def learn(self,num_epochs:int):
        best_val_loss = float("inf")
        train_losses=[]
        val_losses=[]
        for epoch in tqdm(range(num_epochs)):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate(self.val_loader,"val epoch")
            t_l=train_loss["training epoch loss"]
            v_l=val_loss["val epoch loss"]
            train_losses.append(t_l)
            val_losses.append(v_l)

            print(f"\n Epoch {epoch + 1:03d} | train loss: {t_l:.6f} | val loss: {v_l:.6f}")
            if self.scheduler is not None:
                # for ReduceLROnPlateau
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_l)
                else:
                    self.scheduler.step()
            if v_l < best_val_loss:
                best_val_loss = v_l
                torch.save(self.model.state_dict(), "best_cae.pt")

        return {"train epoch loss": train_losses, "val epoch loss": val_losses}

    def test(self):
        best_model_path=Path("best_cae.pt")
        if not best_model_path.exists():
            raise FileNotFoundError(f"best_model_path not found, first train the model")
        self.model.load_state_dict(torch.load("best_cae.pt", map_location=self.device))
        test_loss = self.evaluate(self.test_loader, "test")
        tl=test_loss["test loss"]
        print(f"Test loss: {tl:.6f}")
        return test_loss