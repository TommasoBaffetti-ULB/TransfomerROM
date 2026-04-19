lookahead optimizer: https://arxiv.org/abs/1907.08610

lr shcedulers: https://github.com/Axect/pytorch-scheduler

## Baseline training (no CAE + no Transformer)

To train/test only a baseline model from `ArtFire/DL/Models/Baselines.py` use:

```bash
python baseline_main.py
```

Default baseline model is `unet` (configured in `configs/model.yaml` under `Baseline.model_name`).

You can switch architecture without changing code:

```bash
BASELINE_MODEL=unet python baseline_main.py
BASELINE_MODEL=resnet python baseline_main.py
BASELINE_MODEL=fno2d python baseline_main.py
BASELINE_MODEL=fno3d python baseline_main.py
```

For SLURM execution of U-Net baseline:

```bash
sbatch UNET_RUN.sh
```
