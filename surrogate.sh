#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-sml-staff
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4000
#SBATCH -t 24:00:00
#SBATCH -o /nfs/data_chaos/dbanerjee/output/finetune/second_surrogate_emb2.out
#SBATCH -e /nfs/data_chaos/dbanerjee/error/finetune/second_surrogate_emb2.err

pwd

cd /nfs/data_chaos/dbanerjee/my_data/R2Gen
python="/nfs/data_chaos/dbanerjee/anaconda3/envs/surrogate/bin/python3.6"

$python second_surrogate.py