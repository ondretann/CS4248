#!/bin/bash
#SBATCH --job-name=squad-full-eval
#SBATCH --output=full_eval_%j.out
#SBATCH --error=full_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source venv/bin/activate

echo "=================================================="
echo "Step 1: Generate Predictions"
echo "=================================================="

python3 scripts/evaluate.py \
  --model_path ./models/deberta-v3-squad \
  --validation_file dev-v1.1.json \
  --batch_size 64 \
  --save_predictions predictions.json \
  --output_file results.json

echo ""
echo "=================================================="
echo "Step 2: Official SQuAD Evaluation"
echo "=================================================="

python3 evaluate-v2.0.py dev-v1.1.json predictions.json -o official_results.json

echo ""
echo "=================================================="
echo "Final Results"
echo "=================================================="
cat official_results.json
echo ""
echo "=================================================="

