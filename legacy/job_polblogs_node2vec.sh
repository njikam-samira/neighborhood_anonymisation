#!/bin/bash
#SBATCH --job-name=polblogs_node2vec_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/polblogs_node2vec_%j.out
#SBATCH --error=logs/polblogs_node2vec_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

set -euo pipefail

DATASET_LABEL="PolBlogs"
DATASET_ARG="polblogs"
DATASET_FILE_SLUG="polblogs"
MODEL_KEY="node2vec"

# HIKDA_MAX_NODES derive du nombre de noeuds uniques observes dans data/polblogs.pairs.
HIKDA_MAX_NODES=1224
DATA_PAIRS="data/polblogs.pairs"
DATA_FALLBACK_CANDIDATES=("secGraph/data/polblogs.pairs" "kdld/data/polblogs.pairs" "umga/data/polblogs.pairs")
DATA_GRAPH_ASSET_REQUIRED=1
DATA_GRAPH_ASSET="data/polblogs.gpickle"
DATA_GRAPH_ASSET_FALLBACK_CANDIDATES=("kdld/data/polblogs.gpickle")

source "$(dirname "${BASH_SOURCE[0]}")/_classification_job_common.sh"
