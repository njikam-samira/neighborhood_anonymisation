#!/bin/bash
#SBATCH --job-name=enron_node2vec_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/enron_node2vec_%j.out
#SBATCH --error=logs/enron_node2vec_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

set -euo pipefail

DATASET_LABEL="Enron"
DATASET_ARG="enron"
DATASET_FILE_SLUG="enron"
MODEL_KEY="node2vec"

# HIKDA_MAX_NODES derive du nombre de noeuds uniques observes dans data/Enron.pairs.
HIKDA_MAX_NODES=36692
DATA_PAIRS="data/Enron.pairs"
DATA_FALLBACK_CANDIDATES=("secGraph/data/Enron.pairs" "kdld/data/Enron.pairs" "umga/data/Enron.pairs")
DATA_GRAPH_ASSET_REQUIRED=1
DATA_GRAPH_ASSET="data/Enron.gpickle"
DATA_GRAPH_ASSET_FALLBACK_CANDIDATES=("kdld/data/Enron.gpickle")

source /scratch/rdongmo/IDP-LS/k-Anonymization/legacy/_classification_job_common.sh
