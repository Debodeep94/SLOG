#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024
#SBATCH --job-name=reg_plots
#SBATCH --output=output/only_find/surr_train_only_find.txt
#SBATCH --error=error/only_find/surr_train_only_find.err
#SBATCH -N 1

python main_test.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /home/debodeep.banerjee/R2Gen/data/mimic/only_findings_split.json \
--dataset_name mimic_cxr \
--max_seq_length 40 \
--threshold 10 \
--batch_size 512 \
--epochs 30  \
--step_size 1 \
--gamma 0.8 \
--seed 456788 \
--n_gpu 1 \
--load /home/debodeep.banerjee/R2Gen/results/mimic_cxr/only_find/best_model.pth
