#!/bin/bash
#SBATCH --job-name=pubmed_node2vec_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/pubmed_node2vec_%j.out
#SBATCH --error=logs/pubmed_node2vec_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

set -euo pipefail

DATASET_LABEL="Pubmed"
DATASET_ARG="pubmed"
DATASET_FILE_SLUG="pubmed"
MODEL_KEY="node2vec"

# HIKDA_MAX_NODES derive du nombre de noeuds uniques observes dans data/pubmed.pairs.
HIKDA_MAX_NODES=19717
DATA_PAIRS="data/pubmed.pairs"
DATA_FALLBACK_CANDIDATES=("secGraph/data/pubmed.pairs" "kdld/data/pubmed.pairs" "umga/data/pubmed.pairs")
DATA_GRAPH_ASSET_REQUIRED=0
DATA_GRAPH_ASSET=""
DATA_GRAPH_ASSET_FALLBACK_CANDIDATES=()

source "$(dirname "${BASH_SOURCE[0]}")/_classification_job_common.sh"
