#!/bin/bash
#SBATCH --out=job.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=4096
#SBATCH --account=IscrC_HeR-T            ##using "saldo -b" command
#SBATCH --partition=lrd_all_serial       ##using "sinfo|grep serial" command
#SBATCH --gres=gres:tmpfs:10g

cd /leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Retraining/models
rsync  -PravzHS /Users/WilliamLiu/HeR_T_retaining/models/model.safetensors wliu0000@login.leonardo.cineca.it:/leonardo_work/IscrC_HeR-T/weiwei/HeR-T-Retraining/models  