#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name=R2Gen
#SBATCH -t 2-00
#SBATCH --output=output/imp_n_find/train_mimic.txt
#SBATCH --error=error/imp_n_find/train_mimic.err
#SBATCH -N 1

python main_train.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path ./R2Gen/data/mimic/imp_n_find_corrected.json \
--dataset_name mimic_cxr \
--max_seq_length 70 \
--threshold 10 \
--batch_size 256 \
--early_stop 20 \
--epochs 30 \
--save_dir ./R2Gen/results/mimic_cxr \
--step_size 1 \
--gamma 0.8 \
--num_workers 8 \
--n_gpu 1 \
--beam_size 3 \
--seed 456789 \