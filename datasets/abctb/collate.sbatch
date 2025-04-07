#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=32000
#SBATCH --partition=hmem
#SBATCH --time=01:00:00

module purge
source venv/bin/activate
python3 collate.py
