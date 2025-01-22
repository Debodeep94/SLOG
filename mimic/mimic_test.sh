#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048
#SBATCH --job-name=E70
#SBATCH --output=output/imp_n_find/gen_embed.txt
#SBATCH --error=error/imp_n_find/gen_embed.err
#SBATCH -N 1

python main_test.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /R2Gen/data/mimic/imp_n_find_corrected.json \
--dataset_name mimic_cxr \
--max_seq_length 70 \
--threshold 10 \
--batch_size 512 \
--epochs 30  \
--step_size 1 \
--gamma 0.8 \
--seed 456788 \
--load ./R2Gen/results/mimic_cxr/best_model_full_no_gumbel_70.pth \
--num_workers 8 \
--n_gpu 1 \
--beam_size 2 \
--seed 456789 \