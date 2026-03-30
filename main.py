# ARTFIRE IMPORT
from ArtFire.utils.config import load_data_config, load_model_config, load_train_config
from ArtFire.DL.Models.CAE import CAE, ConvEncoder, ConvDecoder
from ArtFire.Data.CAEDataset import CAEDataset
from ArtFire.Data.ForecastDataset import ForecastDataset
from ArtFire.DL.Optimization.optimizers import build_optimizer, build_parameter_groups
from ArtFire.DL.Training.CAETrainer import CAETrainer
from ArtFire.DL.Training.ForecasterTrainer import ForecasterTrainer
from ArtFire.DL.Models.Forecast import ARTransformerForecaster
from ArtFire.utils.save import save_json

# PYTORCH IMPORT
from pytorch_scheduler.scheduler.polynomial import PolynomialScheduler
from pytorch_scheduler.base.warmup import WarmupScheduler
from torch.utils.data import DataLoader
import torch.nn as nn

# Others
from pathlib import Path

def main():
    print("Loading config...")
    data_config = load_data_config()
    model_config = load_model_config()
    train_config = load_train_config()

    print("Loading CAE data...")
    cae_config=data_config["CAE"]
    data_path = cae_config["data_path"]
    cae_data_split=cae_config["split"]


    train_dataset = CAEDataset(
        npy_path=data_path,
        split=cae_data_split,
        mode="train",
        normalize=cae_config["normalize"],
    )

    val_dataset = CAEDataset(
        npy_path=data_path,
        split=cae_data_split,
        mode="val",
        normalize=cae_config["normalize"],
        stats=train_dataset.get_stats(),
    )
    test_dataset = CAEDataset(
        npy_path=data_path,
        split=cae_data_split,
        mode="test",
        normalize=cae_config["normalize"],
        stats=train_dataset.get_stats(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cae_config["train_batch_size"],
        shuffle=True,
        num_workers=cae_config["loader"]["loaders_num_workers"],
        pin_memory= cae_config["loader"]["pin_memory"],
        persistent_workers= cae_config["loader"]["persistent_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cae_config["val_batch_size"],
        shuffle=True,
        num_workers=cae_config["loader"]["loaders_num_workers"],
        pin_memory=cae_config["loader"]["pin_memory"],
        persistent_workers=cae_config["loader"]["persistent_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cae_config["test_batch_size"],
        shuffle=True,
        num_workers=cae_config["loader"]["loaders_num_workers"],
        pin_memory=cae_config["loader"]["pin_memory"],
        persistent_workers=cae_config["loader"]["persistent_workers"],
    )

    print("Initializing CAE model...")

    CAEenc = ConvEncoder(model_config["CAE"]["spatial_conv_config"], model_config["CAE"]["temporal_conv_config"])
    CAEdec = ConvDecoder(model_config["CAE"]["spatial_tconv_config"], model_config["CAE"]["temporal_tconv_config"],
                         model_config["CAE"]["no_flatten_dim"])

    CAEmodel = CAE(CAEenc, CAEdec).to(train_config["CAE"]["device"])

    print("Setting up optimizer and scheduler...")

    parameters_groups=build_parameter_groups(CAEmodel, lr=train_config["CAE"]["lr"],
                                             weight_decay=train_config["CAE"]["weight_decay"]
                                             )

    cae_optimizer = build_optimizer(params=parameters_groups[1],
                                config={
                                    "name": train_config["CAE"]["optimizer"],
                                    "use_lookahead": train_config["CAE"]["use_lookahead"],
                                }
                                )
    scheduler = PolynomialScheduler(optimizer=cae_optimizer, total_steps=train_config["CAE"]["n_epochs"],
                                    power=train_config["CAE"]["scheduler"]["power"], min_lr=train_config["CAE"]["scheduler"]["min_lr"])
    cae_warmup_scheduler = WarmupScheduler(cae_optimizer, scheduler, warmup_steps=train_config["CAE"]["scheduler"]["warmup_steps"],
                                       warmup_type=train_config["CAE"]["scheduler"]["warmup_type"])

    print("Setting up the CAE Trainer...")

    cae_trainer = CAETrainer(model=CAEmodel,
                             loaders=[train_loader, val_loader, test_loader],
                             optimizer=cae_optimizer,
                             criterion=nn.MSELoss(),
                             scheduler=cae_warmup_scheduler,
                             device=train_config["CAE"]["device"],
                             gradient_clip=train_config["CAE"]["gradient_clip"],
                             )

    print(f"learning for {train_config['CAE']['n_epochs']} epochs")

    cae_train_results=cae_trainer.learn(num_epochs=train_config["CAE"]["n_epochs"])
    saving_folder=Path(train_config["CAE"]["saving_folder"])
    saving_folder.mkdir(parents=True, exist_ok=True)
    save_json(cae_train_results,saving_folder / "train_results.json")


    print("Evaluating CAE model...")
    cae_test_results=cae_trainer.test()
    save_json(cae_test_results,saving_folder / "test_result.json")

    print("Setting up Transformer forecaster data...")
    trans_data_config = data_config["Transformer"]
    full_cae_dataset = CAEDataset(
        npy_path=data_path,
        split=(1.0, 0.0, 0.0),  # IMPORTANT → full dataset
        mode="train",
        normalize=cae_config["normalize"],
    )

    full_cae_dataloader = DataLoader(
        full_cae_dataset,
        batch_size=trans_data_config["cae_dataloader"]["batch_size"],
        shuffle=False,  # do not shuffle for time series
        num_workers= trans_data_config["cae_dataloader"]["loaders_num_workers"],
        pin_memory= trans_data_config["cae_dataloader"]["pin_memory"],
        persistent_workers =  trans_data_config["cae_dataloader"]["persistent_workers"]
    )

    train_forecast = ForecastDataset(
        cae_dataset=full_cae_dataloader,
        encoder=CAEmodel.ConvEncoder,
        split=trans_data_config["split"],
        mode="train",
        horizon=trans_data_config["horizon"],
        normalize=trans_data_config["normalize"],
    )

    val_forecast = ForecastDataset(
        cae_dataset=full_cae_dataloader,
        encoder=CAEmodel.ConvEncoder,
        split=trans_data_config["split"],
        mode="val",
        horizon=trans_data_config["horizon"],
        normalize=trans_data_config["normalize"],
        stats=train_forecast.get_stats(),
    )

    test_forecast = ForecastDataset(
        cae_dataset=full_cae_dataloader,
        encoder=CAEmodel.ConvEncoder,
        split=trans_data_config["split"],
        mode="test",
        horizon=trans_data_config["horizon"],
        normalize=trans_data_config["normalize"],
        stats=train_forecast.get_stats(),
    )

    train_dataloader = DataLoader(
        train_forecast,
        batch_size=trans_data_config["train_batch_size"],
        shuffle=False,  # do not shuffle for time series
        num_workers=trans_data_config["loader"]["loaders_num_workers"],
        pin_memory=trans_data_config["loader"]["pin_memory"],
        persistent_workers=trans_data_config["loader"]["persistent_workers"]
    )

    val_dataloader = DataLoader(
        val_forecast,
        batch_size=trans_data_config["val_batch_size"],
        shuffle=False,  # do not shuffle for time series
        num_workers=trans_data_config["loader"]["loaders_num_workers"],
        pin_memory=trans_data_config["loader"]["pin_memory"],
        persistent_workers=trans_data_config["loader"]["persistent_workers"]
    )

    test_dataloader = DataLoader(
        test_forecast,
        batch_size=trans_data_config["test_batch_size"],
        shuffle=False,  # do not shuffle for time series
        num_workers=trans_data_config["loader"]["loaders_num_workers"],
        pin_memory=trans_data_config["loader"]["pin_memory"],
        persistent_workers=trans_data_config["loader"]["persistent_workers"]
    )
    print("Initializing the Transformer model...")
    model_forecast = ARTransformerForecaster(
        n_tokens=model_config["Transformer"]["n_tokens"],
        token_dim=model_config["Transformer"]["token_dim"],
        d_model=model_config["Transformer"]["d_model"],
        n_heads=model_config["Transformer"]["n_heads"],
        n_layers=model_config["Transformer"]["n_layers"],
        mlp_config=model_config["Transformer"]["mlp_config"],
        dropout=model_config["Transformer"]["dropout"],
        use_residual=model_config["Transformer"]["use_residual"],
    ).to(train_config["Transformer"]["device"])

    print("Setting up optimizer and scheduler...")

    parameters_groups=build_parameter_groups(model_forecast, lr=train_config["Transformer"]["lr"],
                                             weight_decay=train_config["Transformer"]["weight_decay"]
                                             )

    trans_optimizer = build_optimizer(params=parameters_groups[1],
                                config={
                                    "name": train_config["Transformer"]["optimizer"],
                                    "use_lookahead": train_config["Transformer"]["use_lookahead"],
                                }
                                )
    scheduler = PolynomialScheduler(optimizer=trans_optimizer,
                                    total_steps=train_config["Transformer"]["n_epochs"],
                                    power=train_config["Transformer"]["scheduler"]["power"],
                                    min_lr=train_config["Transformer"]["scheduler"]["min_lr"])
    trans_warmup_scheduler = WarmupScheduler(trans_optimizer, scheduler, warmup_steps=train_config["Transformer"]["scheduler"]["warmup_steps"],
                                       warmup_type=train_config["Transformer"]["scheduler"]["warmup_type"])

    print("Setting up the Transformer Trainer...")

    trans_trainer=ForecasterTrainer(model=model_forecast,
                                    loaders=[train_dataloader,val_dataloader,test_dataloader],
                                    optimizer=trans_optimizer,
                                    scheduler=trans_warmup_scheduler,
                                    device=train_config["Transformer"]["device"],
                                    gradient_clip=train_config["Transformer"]["gradient_clip"],
                                    )

    print(f"learning for {train_config['Transformer']['n_epochs']} epochs")

    trans_train_results = trans_trainer.learn(num_epochs=train_config["Transformer"]["n_epochs"])
    saving_folder = Path(train_config["Transformer"]["saving_folder"])
    saving_folder.mkdir(parents=True, exist_ok=True)
    save_json(trans_train_results, saving_folder / "train_results.json")

    print("Evaluating Transformer model...")
    trans_test_results = trans_trainer.test()
    save_json(trans_test_results, saving_folder / "test_result.json")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
