#!/bin/bash
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1
#SBATCH --job-name=l1
#SBATCH --output=output/find_or_imp/sur_only_find_0_0.txt
#SBATCH --error=error/find_or_imp/sur_only_find_0_0.err
#SBATCH -N 1


python main_surrogate.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path /home/debodeep.banerjee/R2Gen/data/mimic/find_or_imp_split.json \
--dataset_name mimic_cxr \
--max_seq_length 40 \
--threshold 10 \
--batch_size 512 \
--surr_weight 0.01 \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--surrogate_model /home/debodeep.banerjee/R2Gen/surrogate/surr_lin_reg_split_find_or_imp.pt \
--load /home/debodeep.banerjee/R2Gen/results/mimic_cxr/find_or_imp/finetuned_best_CE_surr_1_0_sr0_0.pth
