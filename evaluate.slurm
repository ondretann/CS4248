#!/bin/bash
#SBATCH --job-name=squad-eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

python3 evaluate-v2.0.py dev-v1.1.json predictions.json -o official_results.json

echo "Official evaluation results saved to official_results.json"
cat official_results.json

