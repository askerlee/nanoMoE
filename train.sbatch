#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=256G
#SBATCH -p testqueue
#SBATCH -A aicloud-testgroup
#SBATCH -t 18:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
python train.py config/train_nano_moe.py --save_ckpt_every_n_evals=50 --ckpt_prefix="$SLURM_JOB_ID" "$@"
