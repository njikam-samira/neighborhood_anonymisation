#!/bin/bash
#SBATCH --job-name=citeseer-k-sweep-lp
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-citeseer-k-sweep-%j.out

echo "--- START ---"
date
python run_cora_k_sweep_link_prediction.py --dataset citeseer --input data/citeseer.pairs --output-dir resultat/citeseer_link_prediction_k_sweep
echo "--- END ---"
date

