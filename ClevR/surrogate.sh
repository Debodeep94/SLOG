#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name=fs
#SBATCH --output=output/quality_checker_long_new_exp.txt
#SBATCH --error=error/quality_checker_long_new_exp.err
#SBATCH -N 1
pwd
python quality_checker.py --patience 20