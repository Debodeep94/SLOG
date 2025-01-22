#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH -t 2-00
#SBATCH --job-name=mimic0.0
#SBATCH --output=./R2Gen/output/imp_n_find/latest_2surrs/ft_lam0.0.txt
#SBATCH --error=./R2Gen/error/imp_n_find/latest_2surrs/ft_lam0.0.err
#SBATCH -N 1

python main_finetune.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path ./R2Gen/data/mimic/imp_n_find_corrected.json \
--dataset_name mimic_cxr \
--max_seq_length 70 \
--threshold 10 \
--batch_size 64 \
--epochs 10 \
--save_dir ./R2Gen/results/mimic_cxr/imp_n_find/latest_2surrs \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--surr_weight 0.0 \
--llm_weight 1 \
--surrogate_model ./R2Gen/surrogate/sur2_best_model.pt \
--num_workers 12 \
--n_gpu 1 \
--beam_size 2 \
--surrogate_identity 1 \
--monitor_metric BLEU_4 \
--load ./R2Gen/results/mimic_cxr/best_model_full_no_gumbel_70.pth