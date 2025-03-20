#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=econv-abctb-True
#SBATCH --partition=falcon,gecko
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
source /etc/profile.d/modules.sh
module load CUDA
source ../../../venv/bin/activate
srun python3 econv.py abtcb True