#!/bin/bash
#SBATCH --job-name=cora_gat_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/cora_gat_%j.out
#SBATCH --error=logs/cora_gat_%j.err
#SBATCH --time=120:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

set -euo pipefail

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-manual_${RUN_TIMESTAMP}}

PROJECT_DIR=$(pwd)
DATASET_LABEL="Cora"
DATASET_ARG="cora"
MODEL="gat"
K_VALUES="2,5,10,50"
SEEDS="42,123,2024"
METHODS="original,Ange_Original,Ange_Modifie_NCC,Zhou_Pei,1HiKDA"
METRICS="accuracy,macro_f1,micro_f1"
UTILITY_RATIOS="accuracy_utility_ratio,macro_f1_utility_ratio,micro_f1_utility_ratio"
GLOBAL_UTILITY_SCORE_DEF="mean(accuracy_utility_ratio,macro_f1_utility_ratio,micro_f1_utility_ratio)"

RESULT_DIR="resultat/cora_gat_classification_k_sweep"
PLOTS_DIR="plots_cora_gat"
ARCHIVE_DIR="${RESULT_DIR}/archives"
LOG_FILE="logs/cora_gat_${JOB_ID}_${RUN_TIMESTAMP}.log"

SCRIPT="run_cora_gat_classification_k_sweep.py"
DATA_PAIRS="data/cora.pairs"
DATA_FALLBACK_CANDIDATES=(
  "secGraph/data/cora.pairs"
  "kdld/data/cora.pairs"
  "umga/data/cora.pairs"
)

REQUIRED_FILES=(
  "${SCRIPT}"
)

echo "========================================================"
echo "JOB Cora --- GAT Classification"
echo "Job ID   : ${JOB_ID}"
echo "Noeud    : $(hostname)"
echo "Demarre  : $(date)"
echo "Dossier  : ${PROJECT_DIR}"
echo "========================================================"

module load python/3.11.7
source /scratch/rdongmo/IDP-LS/idp_env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs "${RESULT_DIR}" "${PLOTS_DIR}" "${ARCHIVE_DIR}"

if [ ! -f "${DATA_PAIRS}" ]; then
    echo "[INFO] ${DATA_PAIRS} absent. Tentative de recuperation depuis le projet..."
    for candidate in "${DATA_FALLBACK_CANDIDATES[@]}"; do
        if [ -f "${candidate}" ]; then
            mkdir -p "$(dirname "${DATA_PAIRS}")"
            cp "${candidate}" "${DATA_PAIRS}"
            echo "  OK : copie ${candidate} -> ${DATA_PAIRS}"
            break
        fi
    done
fi

if [ ! -f "${DATA_PAIRS}" ]; then
    echo "[WARN] ${DATA_PAIRS} introuvable. Le script Python tentera le fallback Planetoid."
fi

# Sauvegarde de la configuration du job
cat > "${RESULT_DIR}/config_${JOB_ID}.txt" <<CONFIG_EOF
job_id=${JOB_ID}
dataset=${DATASET_LABEL}
dataset_arg=${DATASET_ARG}
model=${MODEL}
k_values=${K_VALUES}
seeds=${SEEDS}
methods=${METHODS}
metrics=${METRICS}
utility_ratios=${UTILITY_RATIOS}
global_utility_score=${GLOBAL_UTILITY_SCORE_DEF}
project_dir=${PROJECT_DIR}
result_dir=${RESULT_DIR}
plots_dir=${PLOTS_DIR}
start_date=$(date)
python_script=${SCRIPT}
hidden_channels=8
heads=8
dropout=0.6
learning_rate=0.005
weight_decay=5e-4
epochs=300
early_stopping=true
CONFIG_EOF

cp "$0" "${RESULT_DIR}/job_cora_gat_${JOB_ID}.sh" 2>/dev/null || true

# Verification des fichiers requis
echo "[CHECK] Fichiers requis :"
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  OK : $f"
    else
        echo "  MANQUANT : $f"
        echo "[ERREUR] Fichier requis absent. Le job s'arrete proprement."
        exit 1
    fi
done
echo ""

echo "Parametres :"
echo "  dataset     : ${DATASET_LABEL}"
echo "  modele      : ${MODEL}"
echo "  k-values    : ${K_VALUES}"
echo "  seeds       : ${SEEDS}"
echo "  methodes    : ${METHODS}"
echo "  metriques   : ${METRICS}"
echo "  ratios      : ${UTILITY_RATIOS}"
echo "  global score: ${GLOBAL_UTILITY_SCORE_DEF}"
echo "  hidden      : 8"
echo "  heads       : 8"
echo "  dropout     : 0.6"
echo "  lr          : 0.005"
echo "  wd          : 5e-4"
echo "  epochs      : 300"
echo "  early stop  : active si supporte"
echo "  data pairs  : ${DATA_PAIRS} ($( [ -f "${DATA_PAIRS}" ] && echo present || echo missing ))"
echo "  resultats   : ${RESULT_DIR}"
echo "  graphiques  : ${PLOTS_DIR}"
echo "  log         : ${LOG_FILE}"
echo ""

# Environnement Python
python3 - <<PY > "${RESULT_DIR}/environment_${JOB_ID}.txt"
import sys
import subprocess

print("Python:", sys.version)
for pkg in ["networkx", "numpy", "pandas", "scikit-learn", "torch", "torch_geometric"]:
    mod_name = "sklearn" if pkg == "scikit-learn" else pkg
    try:
        m = __import__(mod_name)
        print(pkg, getattr(m, "__version__", "version inconnue"))
    except Exception as e:
        print(pkg, "NON DISPONIBLE", str(e))

try:
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    print("\n--- pip freeze ---")
    print(freeze)
except Exception as e:
    print("pip freeze impossible:", e)
PY

echo "========================================================"
echo "PIPELINE Cora GAT"
echo "Demarre : $(date)"
echo "========================================================"

python3 -u "${SCRIPT}" \
    --dataset "${DATASET_ARG}" \
    --k-values "${K_VALUES}" \
    --seeds "${SEEDS}" \
    --methods "${METHODS}" \
    --output-dir "${RESULT_DIR}" \
    --plots-dir "${PLOTS_DIR}" \
    --hidden-channels 8 \
    --heads 8 \
    --dropout 0.6 \
    --learning-rate 0.005 \
    --weight-decay 5e-4 \
    --epochs 300 \
    --early-stopping \
    --save-report \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================================"
echo "COLLECTE DES RESULTATS"
echo "========================================================"

cp "${LOG_FILE}" "${RESULT_DIR}/" 2>/dev/null || true
find "${RESULT_DIR}" "${PLOTS_DIR}" -maxdepth 3 -type f | sort > "${RESULT_DIR}/generated_files_${JOB_ID}.txt" || true

tar -czf "${ARCHIVE_DIR}/cora_gat_results_${JOB_ID}.tar.gz" \
    -C "${RESULT_DIR}" . \
    --exclude="archives/*.tar.gz" 2>/dev/null || true

echo "JOB TERMINE : $(date)"
echo "Job ID      : ${JOB_ID}"
echo ""
echo "Resultats attendus :"
echo "  ${RESULT_DIR}/results_cora_gat_classification.csv"
echo "  ${RESULT_DIR}/results_cora_gat_classification_summary.csv"
echo "  ${RESULT_DIR}/results_cora_gat_classification_report.pdf ou .html"
echo "  ${PLOTS_DIR}/accuracy_by_k_and_method.png"
echo "  ${PLOTS_DIR}/macro_f1_by_k_and_method.png"
echo "  ${PLOTS_DIR}/micro_f1_by_k_and_method.png"
echo "  ${PLOTS_DIR}/utility_score_by_k_and_method.png"
echo ""
echo "Fichiers generes :"
cat "${RESULT_DIR}/generated_files_${JOB_ID}.txt" || true
echo "========================================================"
