#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=clvt_r2genTr
#SBATCH --output=output/trainLONG_new.txt
#SBATCH --error=error/trainLONG_new.err
#SBATCH -N 1

python main_train.py \
--image_dir /home/debodeep.banerjee/synthetic/clevr_images \
--ann_path  /home/debodeep.banerjee/synthetic/clevr_data_long.json \
--dataset_name mimic_cxr \
--max_seq_length 80 \
--threshold 10 \
--batch_size 256 \
--epochs 100 \
--save_dir /home/debodeep.banerjee/model_weights/clevr/version/rel_mem/ \
--monitor_metric PRECISION_MICRO \
--step_size 1 \
--gamma 0.8 \
--num_workers 12 \
--n_gpu 1 \
--beam_size 2 \
--seed 456789 \




