#!/bin/bash
#SBATCH --job-name=cora-k-sweep-lp
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-cora-k-sweep-%j.out

echo "--- START ---"
date
python run_cora_k_sweep_link_prediction.py
echo "--- END ---"
date

