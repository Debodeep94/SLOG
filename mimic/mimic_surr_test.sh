#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=l_10_0
#SBATCH --output=output/imp_n_find/test_pretrained.txt
#SBATCH --error=error/imp_n_find/test_pretrained.err
#SBATCH -N 1


python main_surrogate.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /home/debodeep.banerjee/R2Gen/data/mimic/imp_n_find_split.json \
--dataset_name mimic_cxr \
--max_seq_length 40 \
--threshold 10 \
--batch_size 512 \
--surr_weight 111.0 \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--num_workers 2 \
--surrogate_model /home/debodeep.banerjee/R2Gen/surrogate/ridge_second_surr_logprobs.pt \
--min_max_scaler /home/debodeep.banerjee/R2Gen/min_max_scalar_params_lps.pt \
--load /home/debodeep.banerjee/R2Gen/results/mimic_cxr/best_model.pth