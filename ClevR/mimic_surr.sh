#!/bin/bash
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64
#SBATCH --job-name=lpre0.005
#SBATCH --output=output/base_infer_long0.005.txt
#SBATCH --error=error/base_infer_long0.005.err
#SBATCH -N 1


python main_surrogate.py \
--image_dir /home/debodeep.banerjee/synthetic/clevr_images \
--ann_path  /home/debodeep.banerjee/synthetic/clevr_data_long.json \
--dataset_name mimic_cxr \
--max_seq_length 80 \
--threshold 105 \
--batch_size 512 \
--surr_weight 10.0 \
--step_size 1 \
--gamma 0.8 \
--beam_size 3 \
--seed 456789 \
--surrogate_model /home/debodeep.banerjee/clevr/R2Gen/surrogate/quality_checker_surrogate_long_caps.pth \
--load /home/debodeep.banerjee/model_weights/clevr/version/rel_mem/finetuned_best_CE_surr_swap_1_9_sr0_005.pth