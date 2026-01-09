#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

python train.py config/train_nano_moe.py "$@"
