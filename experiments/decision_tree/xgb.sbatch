#!/bin/bash
#
#SBATCH --job-name=xgb
#SBATCH --partition=gecko,falcon,eagle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mail-type=END,FAIL

source /etc/profile.d/modules.sh
module load CUDA
source ../../../venv/bin/activate
# python3 -m pip install --upgrade -r ../../requirements.txt
python3 -u xgb.py

