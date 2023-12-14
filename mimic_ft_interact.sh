#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH -t 2-00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=L1
#SBATCH --output=output/only_find/interactive/round_1_more/lam_chex_interact_lam0_1.txt
#SBATCH --error=error/only_find/interactive/round_1_more/lam_chex_interact_lam0_1.err
#SBATCH -N 1

python main_finetune.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /home/debodeep.banerjee/R2Gen/data/mimic/surr_chex_only_find_round1_more_data.json \
--dataset_name mimic_cxr \
--max_seq_length 40 \
--threshold 10 \
--batch_size 32 \
--epochs 10 \
--save_dir /home/debodeep.banerjee/R2Gen/results/mimic_cxr/only_find/interactive/round1_more \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--surr_weight 0.1 \
--llm_weight 1 \
--surrogate_model /home/debodeep.banerjee/R2Gen/surrogate/surr_chex_ridge_reg_split_only_find_r1_more_data.pt \
--min_max_scaler /home/debodeep.banerjee/R2Gen/surr_chex_min_max_only_find_r1_more_data.pt \
--num_workers 4 \
--n_gpu 1 \
--beam_size 3 \
--load /home/debodeep.banerjee/R2Gen/results/mimic_cxr/only_find/best_model.pth

