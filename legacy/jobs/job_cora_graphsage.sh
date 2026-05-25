#!/bin/bash
#SBATCH --job-name=cora_graphsage_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/cora_graphsage_%j.out
#SBATCH --error=logs/cora_graphsage_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-manual_${RUN_TIMESTAMP}}

PROJECT_DIR=$(pwd)
DATASET_LABEL="Cora"
DATASET_ARG="cora"
MODEL="graphsage"
K_VALUES="2,5,10,50"
SEEDS="42,123,2024"
METHODS="original,Ange_Original,Ange_Modifie_NCC,Zhou_Pei,1HiKDA"
METRICS="accuracy,macro_f1,micro_f1"
UTILITY_RATIOS="accuracy_utility_ratio,macro_f1_utility_ratio,micro_f1_utility_ratio"
GLOBAL_UTILITY_SCORE_DEF="mean(accuracy_utility_ratio,macro_f1_utility_ratio,micro_f1_utility_ratio)"

RESULT_DIR="resultat/cora_graphsage_classification_k_sweep"
PLOTS_DIR="plots_cora_graphsage"
LOG_FILE="logs/cora_graphsage_${JOB_ID}_${RUN_TIMESTAMP}.log"

SCRIPT="run_cora_graphsage_classification_k_sweep.py"
DATA_PAIRS="data/cora.pairs"
DATA_FALLBACK_CANDIDATES=("secGraph/data/cora.pairs" "kdld/data/cora.pairs" "umga/data/cora.pairs")

echo "========================================================"
echo "JOB Cora --- GraphSAGE Classification"
echo "Job ID   : ${JOB_ID}"
echo "Noeud    : $(hostname)"
echo "Demarre  : $(date)"
echo "Dossier  : ${PROJECT_DIR}"
echo "========================================================"

module load python/3.11.7
source /scratch/rdongmo/IDP-LS/idp_env/bin/activate

mkdir -p logs "${RESULT_DIR}" "${PLOTS_DIR}"

if [ ! -f "${DATA_PAIRS}" ]; then
  echo "[INFO] ${DATA_PAIRS} absent. Tentative de recuperation..."
  for candidate in "${DATA_FALLBACK_CANDIDATES[@]}"; do
    if [ -f "${candidate}" ]; then
      mkdir -p "$(dirname "${DATA_PAIRS}")"
      cp "${candidate}" "${DATA_PAIRS}"
      echo "  OK : copie ${candidate} -> ${DATA_PAIRS}"
      break
    fi
  done
fi

echo "[CHECK] Fichiers requis :"
for f in "${SCRIPT}"; do
  [ -f "$f" ] && echo "  OK : $f" || echo "  MANQUANT : $f"
done
if [ -f "${DATA_PAIRS}" ]; then
  echo "  OK : ${DATA_PAIRS}"
else
  echo "  MANQUANT : ${DATA_PAIRS} (fallback Planetoid actif dans le script Python)"
fi
echo ""

if [ ! -f "${SCRIPT}" ]; then
  echo "[ERREUR] Script ${SCRIPT} absent. Job interrompu."
  exit 1
fi

echo "Parametres :"
echo "  model       : ${MODEL}"
echo "  dataset     : ${DATASET_LABEL}"
echo "  k-values    : ${K_VALUES}"
echo "  seeds       : ${SEEDS}"
echo "  methodes    : ${METHODS}"
echo "  metriques   : ${METRICS}"
echo "  ratios      : ${UTILITY_RATIOS}"
echo "  score global: ${GLOBAL_UTILITY_SCORE_DEF}"
echo "  data pairs  : ${DATA_PAIRS}"
echo "  repertoire  : ${PROJECT_DIR}"
echo ""

echo "========================================================"
echo "PIPELINE Cora GraphSAGE"
echo "Demarre : $(date)"
echo "========================================================"

python3 -u "${SCRIPT}"   --dataset "${DATASET_ARG}"   --k-values "${K_VALUES}"   --seeds "${SEEDS}"   --methods "${METHODS}"   --output-dir "${RESULT_DIR}"   --plots-dir "${PLOTS_DIR}"   --hidden-channels 64   --dropout 0.5   --learning-rate 0.01   --weight-decay 5e-4   --epochs 200   --early-stopping   --save-report   2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================================"
echo "JOB TERMINE : $(date)"
echo "Job ID      : ${JOB_ID}"
echo ""
echo "Resultats dans :"
echo "  ${RESULT_DIR}/"
echo "  ${PLOTS_DIR}/"
echo "  ${LOG_FILE}"
echo "========================================================"
