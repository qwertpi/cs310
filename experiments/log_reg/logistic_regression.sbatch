#!/bin/bash
#
#SBATCH --job-name=logreg
#SBATCH --partition=parrot,cpu-batch
#SBATCH --ntasks=6
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source /etc/profile.d/modules.sh
source ../../../venv/bin/activate
# python3 -m pip install --upgrade -r ../../requirements.txt
parallel -N 1 -j $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.log python3 logistic_regression.py ::: `cat logreg_arglist.txt`

