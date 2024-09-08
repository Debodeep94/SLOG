#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH -t 2-00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name=clv0.0
#SBATCH --output=/home/debodeep.banerjee/clevr/R2Gen/output/finetune/lam_0.0_long_new.txt
#SBATCH --error=/home/debodeep.banerjee/clevr/R2Gen/error/finetune/lam_0.0_long_new.err
#SBATCH -N 1
python main_finetune.py \
--image_dir /home/debodeep.banerjee/synthetic/clevr_images \
--ann_path  /home/debodeep.banerjee/synthetic/clevr_data_long.json \
--dataset_name mimic_cxr \
--max_seq_length 80 \
--threshold 10 \
--monitor_metric POSITIVE_F1 \
--batch_size 32 \
--epochs 10 \
--save_dir /home/debodeep.banerjee/model_weights/clevr/version/rel_mem \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--surr_weight 0.0 \
--llm_weight 1.0 \
--surrogate_model /home/debodeep.banerjee/clevr/R2Gen/surrogate/quality_checker_surrogate_long_caps_exp.pth \
--num_workers 8 \
--n_gpu 1 \
--beam_size 3 \
--load /home/debodeep.banerjee/model_weights/clevr/version/rel_mem/best_model_full_no_gumbel_80.pth