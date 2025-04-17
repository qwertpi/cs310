#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --job-name=gat
#SBATCH --partition=falcon,gecko
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=48:00:0
#
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source /etc/profile.d/modules.sh
module load CUDA
source ../../../venv/bin/activate
# python3 -m pip install --upgrade -r ../../requirements.txt
srun python3 -u gat.py

