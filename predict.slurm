#!/bin/bash
#SBATCH --job-name=squad-predict
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source venv/bin/activate

python3 scripts/evaluate.py \
  --model_path ./models/deberta-v3-squad \
  --validation_file dev-v1.1.json \
  --batch_size 64 \
  --save_predictions predictions.json \
  --output_file results.json

echo "Predictions saved to predictions.json"
echo "Results saved to results.json"

