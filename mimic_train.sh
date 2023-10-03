#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=train
#SBATCH --output=output/only_find/train.txt
#SBATCH --error=error/only_find/train.err
#SBATCH -N 1

python main_train.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /home/debodeep.banerjee/R2Gen/data/mimic/only_findings_split.json \
--dataset_name mimic_cxr \
--max_seq_length 40 \
--threshold 10 \
--batch_size 512 \
--epochs 100 \
--save_dir /home/debodeep.banerjee/R2Gen/results/mimic_cxr/only_find \
--step_size 1 \
--gamma 0.8 \
--num_workers 8 \
--n_gpu 1 \
--beam_size 3 \
--seed 456789 \




