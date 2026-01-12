#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
python train.py config/train_nano_moe.py --save_ckpt_every_n_evals=10 --ckpt_prefix="$SLURM_JOB_ID" "$@"
