#!/bin/bash
#SBATCH -p long-disi 
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=EmbVan
#SBATCH --output=output/gen_embed1.txt
#SBATCH --error=error/gen_embed1.err
#SBATCH -N 1

python main_test.py \
--image_dir /home/debodeep.banerjee/synthetic/clevr_images \
--ann_path  /home/debodeep.banerjee/synthetic/clevr_data_long.json \
--dataset_name mimic_cxr \
--max_seq_length 80 \
--threshold 10 \
--batch_size 512 \
--epochs 30  \
--step_size 1 \
--gamma 0.8 \
--seed 456788 \
--n_gpu 1 \
--beam_size 3 \
--load /home/debodeep.banerjee/model_weights/clevr/version/rel_mem//best_model_full_no_gumbel_80.pth