#!/bin/bash
#SBATCH --job-name=polblogs_graphsage_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/polblogs_graphsage_%j.out
#SBATCH --error=logs/polblogs_graphsage_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

set -euo pipefail

DATASET_LABEL="PolBlogs"
DATASET_ARG="polblogs"
DATASET_FILE_SLUG="polblogs"
MODEL_KEY="graphsage"

# HIKDA_MAX_NODES derive de torch_geometric.datasets.PolBlogs (1490 noeuds).
HIKDA_MAX_NODES=1490
DATA_PAIRS="data/polblogs.pairs"
DATA_FALLBACK_CANDIDATES=("secGraph/data/polblogs.pairs" "kdld/data/polblogs.pairs" "umga/data/polblogs.pairs")
DATA_GRAPH_ASSET_REQUIRED=0
DATA_GRAPH_ASSET=""
DATA_GRAPH_ASSET_FALLBACK_CANDIDATES=()

source "$(dirname "${BASH_SOURCE[0]}")/_classification_job_common.sh"
