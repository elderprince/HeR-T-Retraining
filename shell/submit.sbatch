#!/bin/bash

#SBATCH --job-name=20250312_cinecaInferenceTest

#SBATCH --nodes=1                    # 1 node
#SBATCH --gres=gpu:1                 # 1 GPUs
#SBATCH --ntasks-per-node=4          # 32 tasks per node
#SBATCH --time=2:00:00               # time limits: 2 hour
#SBATCH --error=outputs/myJob.err    # standard error file
#SBATCH --output=outputs/myJob.out   # standard output file
#SBATCH --account=IscrC_HeR-T        # account name
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --mem=256000                 # memory per node out of 494000MB (481GB)

eval "$(conda shell.bash hook)"
source /leonardo_work/IscrC_HeR-T/weiwei/HeR-T-240807/bin/activate

cd /leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Fine-tuning
srun python /leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Fine-tuning/scripts/inference.py