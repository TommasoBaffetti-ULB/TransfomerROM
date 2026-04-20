#!/bin/bash
#SBATCH --job-name=hpo_bl_%j
#SBATCH --output=logs_hpo_bl/hpo_resnet_%j.out
#SBATCH --error=logs_hpo_bl/hpo_resnet_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tommaso.baffetti@ulb.be

#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

# === User-configurable variables (override with: sbatch --export=ALL,VAR=...) ===
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
VENV_PATH="${VENV_PATH:-/globalsc/ulb/atm/baffetti/envs/artfire_new/bin/activate}"
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
ENTRYPOINT="${ENTRYPOINT:-Baseline_hpo.py}"

mkdir -p "$PROJECT_DIR/logs_hpo_bl"
cd "$PROJECT_DIR"

echo "[$(date)] Job started on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ENTRYPOINT=$ENTRYPOINT"

echo "Activating environment: $VENV_PATH"

if [[ -f "$VENV_PATH" ]]; then
  source "$VENV_PATH"
else
  echo "ERROR: VENV_PATH not found ($VENV_PATH)"
  exit 1
fi

export PYTHONNOUSERSITE=1

which python
which pip

python3 --version
nvidia-smi || true

echo "Python usato:"
which python
python -c "import sys; print(sys.executable)"

echo "Torch test:"
python -c "import torch; print(torch.__version__)"

python3 "$ENTRYPOINT" --baseline-model resnet --n-trials 100 --hpo-epochs 25 --storage sqlite:///hpo.db

echo "[$(date)] Job finished"