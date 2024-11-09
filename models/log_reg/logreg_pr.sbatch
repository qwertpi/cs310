#!/bin/bash
#
#SBATCH --job-name=lr-pr
#SBATCH --partition=parrot
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#
#SBATCH --mail-type=END,FAIL

source /etc/profile.d/modules.sh
source ../../../venv/bin/activate
python3 -m pip install --upgrade -r ../../requirements.txt
parallel -N 2 -j $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.log python3 logistic_regression.py PR mean ::: `cat logreg_arglist.txt`
