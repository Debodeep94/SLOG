#!/bin/bash
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=l_10.0
#SBATCH --output=./R2Gen/output/imp_n_find/test_10.0_new.txt
#SBATCH --error=./R2Gen/error/imp_n_find/test_10.0_new.err
#SBATCH -N 1


python main_surrogate.py \
--image_dir /data/MimicCXR/mimic_images/ \
--ann_path ./R2Gen/data/mimic/imp_n_find_corrected.json \
--dataset_name mimic_cxr \
--max_seq_length 70 \
--threshold 10 \
--batch_size 128 \
--surr_weight 10.0 \
--step_size 1 \
--gamma 0.8 \
--beam_size 2 \
--seed 456789 \
--num_workers 4 \
--n_gpu 1 \
--surrogate_model ./R2Gen/surrogate/sur2_best_model.pt \
--min_max_scaler ./R2Gen/min_max_scalar_params_lps.pt \
--load ./R2Gen/results/mimic_cxr/imp_n_find/latest_2surrs/finetuned_best_CE_surr_swap_1_0_sr10_0.pth