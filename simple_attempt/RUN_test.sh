#!/bin/bash

#SBATCH --job-name=test_ArtFire_%j
#SBATCH --output=/globalsc/ulb/atm/baffetti/research/Transformer/ArtFire_files/test_ArtFire_%j.out
#SBATCH --error=/globalsc/ulb/atm/baffetti/research/Transformer/ArtFire_files/test_ArtFire_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tommaso.baffetti@ulb.be

#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

echo "Start:"
date

module load Python/3.11.3
source /globalsc/ulb/atm/baffetti/envs/artfire/bin/activate

python3 test.py --config test.json

echo "End:"
date