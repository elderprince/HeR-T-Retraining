#!/bin/bash

#SBATCH --job-name=20240703_cinecaTest

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=4          # 32 tasks per node
#SBATCH --time=2:00:00               # time limits: 2 hour
#SBATCH --error=myJob.err            # standard error file
#SBATCH --output=myJob.out           # standard output file
#SBATCH --account=IscrC_HeR-T        # account name
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --mem=256000                 # memory per node out of 494000MB (481GB)

eval "$(conda shell.bash hook)"
conda activate /leonardo_work/IscrC_HeR-T/weiwei/HeR_T

cd /leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Retraining/outputs
python /leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Retraining/scripts/training.py