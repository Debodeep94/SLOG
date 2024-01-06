#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=annotation
#SBATCH --output=output/surrogate_first.txt
#SBATCH --error=error/surrogate_first.err
#SBATCH -N 1

pwd

python first_surrogate.py