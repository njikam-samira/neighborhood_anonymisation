from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.evaluation.classification_k_sweep import main


if __name__ == "__main__":
    main(default_model="node2vec_logreg")
