#!/bin/bash
#SBATCH --job-name=wiki_vote_node2vec_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/wiki_vote_node2vec_%j.out
#SBATCH --error=logs/wiki_vote_node2vec_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

set -euo pipefail

DATASET_LABEL="Wiki-Vote"
DATASET_ARG="wiki-vote"
DATASET_FILE_SLUG="wiki_vote"
MODEL_KEY="node2vec"

# HIKDA_MAX_NODES derive du nombre de noeuds uniques observes dans data/Wiki-Vote.pairs.
HIKDA_MAX_NODES=7115
DATA_PAIRS="data/Wiki-Vote.pairs"
DATA_FALLBACK_CANDIDATES=("secGraph/data/Wiki-Vote.pairs" "kdld/data/Wiki-Vote.pairs" "umga/data/Wiki-Vote.pairs")
DATA_GRAPH_ASSET_REQUIRED=1
DATA_GRAPH_ASSET="data/Wiki-Vote.gpickle"
DATA_GRAPH_ASSET_FALLBACK_CANDIDATES=("kdld/data/Wiki-Vote.gpickle")

source /scratch/rdongmo/IDP-LS/k-Anonymization/legacy/_classification_job_common.sh
