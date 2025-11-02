#!/bin/bash
#SBATCH --job-name=squad-train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

python3 scripts/train.py \
  --train_file train-v1.1.json \
  --validation_file dev-v1.1.json \
  --output_dir ./models/deberta-v3-squad \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --fp16 \
  --logging_steps 100 \
  --save_steps 999999 \
  --save_total_limit 1 \
  --num_proc 4

