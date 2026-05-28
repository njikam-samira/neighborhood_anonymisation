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

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${SLURM_JOB_ID:-manual_${RUN_TIMESTAMP}}"

PROJECT_DIR="$(pwd)"
JOB_SCRIPT_PATH="${BASH_SOURCE[0]}"
JOB_SCRIPT_NAME="$(basename "${JOB_SCRIPT_PATH}")"
DATASET_LABEL="Cora"
DATASET_ARG="cora"
MODEL="node2vec_logreg"
K_VALUES="2,5,10,50"
SEEDS="42,123,2024"
METHODS="original,Ange_Original,Ange_Modifie_NCC,Zhou_Pei,1HiKDA"
METRICS="accuracy,macro_f1,micro_f1"
UTILITY_RATIOS="accuracy_utility_ratio,macro_f1_utility_ratio,micro_f1_utility_ratio"
GLOBAL_UTILITY_SCORE_DEF="mean(accuracy_utility_ratio,macro_f1_utility_ratio,micro_f1_utility_ratio)"

RESULT_DIR="resultat/cora_node2vec_classification_k_sweep"
PLOTS_DIR="plots_cora_node2vec"
LOG_FILE="logs/cora_node2vec_${JOB_ID}_${RUN_TIMESTAMP}.log"
RUN_DIR="${RESULT_DIR}/run_${JOB_ID}_${RUN_TIMESTAMP}"
CONFIG_FILE="${RUN_DIR}/config_${JOB_ID}.txt"
ENV_FILE="${RUN_DIR}/environment_${JOB_ID}.txt"
GIT_CONTEXT_FILE="${RUN_DIR}/git_context_${JOB_ID}.txt"
CHECK_FILE="${RUN_DIR}/required_files_check_${JOB_ID}.txt"
MANIFEST_FILE="${RUN_DIR}/outputs_manifest_${JOB_ID}.txt"
FAILURE_FILE="${RUN_DIR}/failure_report_${JOB_ID}.txt"

SCRIPT="run_cora_node2vec_classification_k_sweep.py"
DATA_PAIRS="data/cora.pairs"
DATA_FALLBACK_CANDIDATES=("secGraph/data/cora.pairs" "kdld/data/cora.pairs" "umga/data/cora.pairs")
REPORT_PDF="${RESULT_DIR}/results_cora_node2vec_classification_report.pdf"
REPORT_HTML="${RESULT_DIR}/results_cora_node2vec_classification_report.html"
DETAILED_CSV="${RESULT_DIR}/results_cora_node2vec_classification.csv"
SUMMARY_CSV="${RESULT_DIR}/results_cora_node2vec_classification_summary.csv"
EXPECTED_PLOTS=(
  "${PLOTS_DIR}/accuracy_by_k_and_method.png"
  "${PLOTS_DIR}/macro_f1_by_k_and_method.png"
  "${PLOTS_DIR}/micro_f1_by_k_and_method.png"
  "${PLOTS_DIR}/utility_score_by_k_and_method.png"
)
PY_CMD=(
  python3 -u "${SCRIPT}"
  --dataset "${DATASET_ARG}"
  --k-values "${K_VALUES}"
  --seeds "${SEEDS}"
  --methods "${METHODS}"
  --output-dir "${RESULT_DIR}"
  --plots-dir "${PLOTS_DIR}"
  --embedding-dim 128
  --walk-length 20
  --context-size 10
  --walks-per-node 10
  --p 1.0
  --q 1.0
  --epochs 100
  --save-report
)

mkdir -p logs "${RESULT_DIR}" "${PLOTS_DIR}" "${RUN_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
  local exit_code=$?
  local line_no="$1"
  local failed_cmd="$2"

  {
    echo "job_id=${JOB_ID}"
    echo "timestamp=$(date -Is)"
    echo "host=$(hostname)"
    echo "line=${line_no}"
    echo "command=${failed_cmd}"
    echo "exit_code=${exit_code}"
    echo "log_file=${LOG_FILE}"
    echo "result_dir=${RESULT_DIR}"
    echo "run_dir=${RUN_DIR}"
    echo "hint=Inspect ${LOG_FILE}, ${ENV_FILE}, ${GIT_CONTEXT_FILE}, ${CHECK_FILE}"
  } > "${FAILURE_FILE}"

  echo ""
  echo "[ERREUR] Echec detecte. Resume ecrit dans: ${FAILURE_FILE}"
  exit "${exit_code}"
}

trap 'on_error "${LINENO}" "${BASH_COMMAND}"' ERR

echo "========================================================"
echo "JOB Cora --- Node2Vec + Logistic Regression Classification"
echo "Job ID   : ${JOB_ID}"
echo "Noeud    : $(hostname)"
echo "Demarre  : $(date)"
echo "Dossier  : ${PROJECT_DIR}"
echo "========================================================"

module load python/3.11.7

ENV_CANDIDATES=(
  "${HOME}/venv_metrics/bin/activate"
  "/scratch/rdongmo/IDP-LS/idp_env/bin/activate"
)

VENV_ACTIVATE_PATH=""
for candidate in "${ENV_CANDIDATES[@]}"; do
  if [ -f "${candidate}" ]; then
    VENV_ACTIVATE_PATH="${candidate}"
    break
  fi
done

if [ -z "${VENV_ACTIVATE_PATH}" ]; then
  echo "[ERREUR] Aucun environnement virtuel trouve. Candidats testes:"
  for candidate in "${ENV_CANDIDATES[@]}"; do
    echo "  - ${candidate}"
  done
  exit 1
fi

source "${VENV_ACTIVATE_PATH}"

echo "Environment active : ${VENV_ACTIVATE_PATH}"
echo "Python utilise     : $(command -v python3)"
echo "Version Python     : $(python3 --version)"

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

missing_required=0
echo "[CHECK] Fichiers requis :" | tee "${CHECK_FILE}"
for f in "${SCRIPT}"; do
  if [ -f "${f}" ]; then
    line="  OK : ${f}"
  else
    line="  MANQUANT : ${f}"
    missing_required=1
  fi
  echo "${line}" | tee -a "${CHECK_FILE}"
done

if [ -f "${DATA_PAIRS}" ]; then
  line="  OK : ${DATA_PAIRS}"
else
  line="  MANQUANT : ${DATA_PAIRS} (fallback Planetoid actif dans le script Python)"
fi
echo "${line}" | tee -a "${CHECK_FILE}"
echo ""

if [ "${missing_required}" -ne 0 ]; then
  echo "[ERREUR] Un fichier requis est absent. Job interrompu."
  exit 1
fi

echo "[CHECK] Dependances Python critiques (Node2Vec) :"
python3 - <<'PY'
import importlib
import sys

missing = []
for module_name in ["torch", "torch_geometric", "numpy", "pandas", "sklearn", "networkx", "matplotlib"]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - runtime check
        missing.append(f"{module_name}: {exc}")

try:
    from torch_geometric.nn import Node2Vec  # noqa: F401
except Exception as exc:  # pragma: no cover - runtime check
    missing.append(f"torch_geometric.nn.Node2Vec: {exc}")

if missing:
    print("[ERREUR] Dependances manquantes ou incompatibles detectees :")
    for item in missing:
        print(f"  - {item}")
    sys.exit(1)

print("[OK] Dependances Node2Vec disponibles.")
PY

echo "Parametres :"
echo "  model       : ${MODEL}"
echo "  dataset     : ${DATASET_LABEL}"
echo "  k-values    : ${K_VALUES}"
echo "  seeds       : ${SEEDS}"
echo "  methodes    : ${METHODS}"
echo "  metriques   : ${METRICS}"
echo "  ratios      : ${UTILITY_RATIOS}"
echo "  score global: ${GLOBAL_UTILITY_SCORE_DEF}"
echo "  data root   : data/"
echo "  data pairs  : ${DATA_PAIRS}"
echo "  report      : ${REPORT_PDF} (ou HTML)"
echo "  log file    : ${LOG_FILE}"
echo "  run dir     : ${RUN_DIR}"
echo ""

{
  echo "job_id=${JOB_ID}"
  echo "run_timestamp=${RUN_TIMESTAMP}"
  echo "hostname=$(hostname)"
  echo "project_dir=${PROJECT_DIR}"
  echo "job_script=${JOB_SCRIPT_PATH}"
  echo "venv_activate=${VENV_ACTIVATE_PATH}"
  echo "python_bin=$(command -v python3)"
  echo "python_version=$(python3 --version 2>&1)"
  echo "model=${MODEL}"
  echo "dataset=${DATASET_LABEL}"
  echo "dataset_arg=${DATASET_ARG}"
  echo "k_values=${K_VALUES}"
  echo "seeds=${SEEDS}"
  echo "methods=${METHODS}"
  echo "metrics=${METRICS}"
  echo "utility_ratios=${UTILITY_RATIOS}"
  echo "global_utility_score=${GLOBAL_UTILITY_SCORE_DEF}"
  echo "result_dir=${RESULT_DIR}"
  echo "plots_dir=${PLOTS_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "command=${PY_CMD[*]}"
} > "${CONFIG_FILE}"

cp "${JOB_SCRIPT_PATH}" "${RUN_DIR}/${JOB_SCRIPT_NAME}"

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "which_python=$(command -v python3)"
  python3 --version
  echo ""
  python3 -m pip --version
  echo ""
  python3 - <<'PY'
import importlib
import platform
import sys

print(f"platform: {platform.platform()}")
print(f"python: {sys.version.replace(chr(10), ' ')}")

for module_name in ["torch", "torch_geometric", "numpy", "pandas", "sklearn", "networkx", "matplotlib"]:
    try:
        mod = importlib.import_module(module_name)
        print(f"{module_name}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{module_name}: MISSING ({exc})")
PY
  echo ""
  echo "# pip freeze"
  python3 -m pip freeze | sort
} > "${ENV_FILE}" 2>&1 || true

{
  echo "timestamp=$(date -Is)"
  echo "pwd=$(pwd)"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  git diff --stat 2>/dev/null || true
} > "${GIT_CONTEXT_FILE}" 2>&1 || true

echo "========================================================"
echo "PIPELINE Cora Node2Vec"
echo "Demarre : $(date)"
echo "========================================================"

"${PY_CMD[@]}"

{
  echo "job_id=${JOB_ID}"
  echo "timestamp=$(date -Is)"
  echo "result_dir=${RESULT_DIR}"
  echo "plots_dir=${PLOTS_DIR}"
  echo "log_file=${LOG_FILE}"
  echo ""
  for f in "${DETAILED_CSV}" "${SUMMARY_CSV}"; do
    if [ -f "${f}" ]; then
      echo "OK: ${f}"
    else
      echo "MISSING: ${f}"
    fi
  done
  if [ -f "${REPORT_PDF}" ]; then
    echo "OK: ${REPORT_PDF}"
  elif [ -f "${REPORT_HTML}" ]; then
    echo "OK: ${REPORT_HTML}"
  else
    echo "MISSING: ${REPORT_PDF} or ${REPORT_HTML}"
  fi
  for p in "${EXPECTED_PLOTS[@]}"; do
    if [ -f "${p}" ]; then
      echo "OK: ${p}"
    else
      echo "MISSING: ${p}"
    fi
  done
  echo ""
  echo "[FILES] Resultat (maxdepth=2):"
  find "${RESULT_DIR}" -maxdepth 2 -type f | sort
} > "${MANIFEST_FILE}"

echo ""
echo "========================================================"
echo "JOB TERMINE : $(date)"
echo "Job ID      : ${JOB_ID}"
echo ""
echo "Resultats dans :"
echo "  ${RESULT_DIR}/"
echo "  ${PLOTS_DIR}/"
echo "  ${LOG_FILE}"
echo "Diagnostics :"
echo "  ${CONFIG_FILE}"
echo "  ${ENV_FILE}"
echo "  ${GIT_CONTEXT_FILE}"
echo "  ${CHECK_FILE}"
echo "  ${MANIFEST_FILE}"
echo "========================================================"
