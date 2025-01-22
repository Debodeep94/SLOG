#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name=fs_nf
#SBATCH --output=./R2Gen/output/second_surrogate_test.txt
#SBATCH --error=./R2Gen/error/second_surrogate_test.err
#SBATCH -N 1

python second_surrogate.py --patience 10