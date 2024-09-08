#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH -t 2-00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=R0_1
#SBATCH --output=output/imp_n_find/with_lps/finetune_lam_0_1.txt
#SBATCH --error=error/imp_n_find/with_lps/finetune_lam_0_1.err
#SBATCH -N 1

python main_finetune.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /home/debodeep.banerjee/R2Gen/data/mimic/imp_n_find_split.json \
--dataset_name mimic_cxr \
--max_seq_length 40 \
--threshold 10 \
--batch_size 32 \
--epochs 10 \
--save_dir /home/debodeep.banerjee/R2Gen/results/mimic_cxr/imp_n_find/logprobs \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--surr_weight 0.1 \
--llm_weight 1 \
--surrogate_model /home/debodeep.banerjee/R2Gen/best_model_surr2_tial.pth \
--num_workers 12 \
--n_gpu 1 \
--beam_size 3 \
--min_max_scaler /home/debodeep.banerjee/R2Gen/min_max_scalar_params_lps.pt \
--load /home/debodeep.banerjee/R2Gen/results/mimic_cxr/imp_n_find/best_model.pth

