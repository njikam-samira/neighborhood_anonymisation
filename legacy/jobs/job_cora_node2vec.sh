#!/bin/bash
#SBATCH --job-name=cora_node2vec_cls
#SBATCH --chdir=/scratch/rdongmo/IDP-LS/k-Anonymization
#SBATCH --output=logs/cora_node2vec_%j.out
#SBATCH --error=logs/cora_node2vec_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal-amd

set -euo pipefail

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-manual_${RUN_TIMESTAMP}}

PROJECT_DIR=$(pwd)
DATASET="cora"
MODEL="node2vec_logreg"
K_VALUES="2,5,10,50"
SEEDS="42,123,2024"
METHODS="original,Ange_Original,Ange_Modifie_NCC,Zhou_Pei,1HiKDA"

RESULT_DIR="resultat/cora_node2vec_classification_k_sweep"
PLOTS_DIR="${RESULT_DIR}/plots"
ARCHIVE_DIR="${RESULT_DIR}/archives"
LOG_FILE="logs/cora_node2vec_${JOB_ID}_${RUN_TIMESTAMP}.log"

SCRIPT="run_cora_node2vec_classification_k_sweep.py"

echo "========================================================"
echo "JOB Cora --- Node2Vec + Logistic Regression"
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

# Sauvegarde de la configuration du job
cat > "${RESULT_DIR}/config_${JOB_ID}.txt" <<CONFIG_EOF
job_id=${JOB_ID}
dataset=${DATASET}
model=${MODEL}
k_values=${K_VALUES}
seeds=${SEEDS}
methods=${METHODS}
project_dir=${PROJECT_DIR}
result_dir=${RESULT_DIR}
plots_dir=${PLOTS_DIR}
start_date=$(date)
python_script=${SCRIPT}
embedding_dim=128
walk_length=20
context_size=10
walks_per_node=10
p=1.0
q=1.0
epochs=100
CONFIG_EOF

cp "$0" "${RESULT_DIR}/job_cora_node2vec_${JOB_ID}.sh" 2>/dev/null || true

# Verification des fichiers requis
echo "[CHECK] Fichiers requis :"
for f in "${SCRIPT}"; do
    [ -f "$f" ] && echo "  OK : $f" || { echo "  MANQUANT : $f"; exit 1; }
done
echo ""

echo "Parametres :"
echo "  dataset     : ${DATASET}"
echo "  modele      : ${MODEL}"
echo "  k-values    : ${K_VALUES}"
echo "  seeds       : ${SEEDS}"
echo "  methodes    : ${METHODS}"
echo "  resultats   : ${RESULT_DIR}"
echo "  graphiques  : ${PLOTS_DIR}"
echo "  log         : ${LOG_FILE}"
echo ""

# Environnement Python
python3 - <<PY > "${RESULT_DIR}/environment_${JOB_ID}.txt"
import sys, subprocess
print("Python:", sys.version)
for pkg in ["networkx", "numpy", "pandas", "sklearn", "torch", "torch_geometric"]:
    try:
        m = __import__(pkg)
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
echo "PIPELINE Cora Node2Vec + Logistic Regression"
echo "Demarre : $(date)"
echo "========================================================"

python3 -u "${SCRIPT}" \
    --dataset "${DATASET}" \
    --k-values "${K_VALUES}" \
    --seeds "${SEEDS}" \
    --methods "${METHODS}" \
    --output-dir "${RESULT_DIR}" \
    --plots-dir "${PLOTS_DIR}" \
    --embedding-dim 128 \
    --walk-length 20 \
    --context-size 10 \
    --walks-per-node 10 \
    --p 1.0 \
    --q 1.0 \
    --epochs 100 \
    --save-report \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================================"
echo "COLLECTE DES RESULTATS"
echo "========================================================"

# Copie du log dans le dossier resultats
cp "${LOG_FILE}" "${RESULT_DIR}/" 2>/dev/null || true

# Resume des fichiers generes
find "${RESULT_DIR}" -maxdepth 3 -type f | sort > "${RESULT_DIR}/generated_files_${JOB_ID}.txt" || true

# Archivage compact des resultats du job
tar -czf "${ARCHIVE_DIR}/cora_node2vec_results_${JOB_ID}.tar.gz" \
    -C "${RESULT_DIR}" . \
    --exclude="archives/*.tar.gz" 2>/dev/null || true

echo "JOB TERMINE : $(date)"
echo "Job ID      : ${JOB_ID}"
echo ""
echo "Resultats attendus :"
echo "  ${RESULT_DIR}/results_cora_node2vec_classification.csv"
echo "  ${RESULT_DIR}/results_cora_node2vec_classification_summary.csv"
echo "  ${RESULT_DIR}/results_cora_node2vec_classification_report.pdf ou .html"
echo "  ${PLOTS_DIR}/accuracy_by_k_and_method.png"
echo "  ${PLOTS_DIR}/macro_f1_by_k_and_method.png"
echo "  ${PLOTS_DIR}/micro_f1_by_k_and_method.png"
echo "  ${PLOTS_DIR}/utility_score_by_k_and_method.png"
echo ""
echo "Fichiers generes :"
cat "${RESULT_DIR}/generated_files_${JOB_ID}.txt" || true
echo "========================================================"
