#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=reg
#SBATCH --output=/home/debodeep.banerjee/R2Gen/output/imp_n_find/surr_training_samp_lps.txt
#SBATCH --error=/home/debodeep.banerjee/R2Gen/error/imp_n_find/surr_training_samp_lps.err
#SBATCH -N 1

python main_test.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /home/debodeep.banerjee/R2Gen/data/mimic/imp_n_find_split.json \
--dataset_name mimic_cxr \
--max_seq_length 40 \
--threshold 10 \
--batch_size 512 \
--epochs 30  \
--step_size 1 \
--gamma 0.8 \
--seed 456788 \
--num_workers 12 \
--n_gpu 1 \
--load /home/debodeep.banerjee/R2Gen/results/mimic_cxr/imp_n_find/best_model.pth