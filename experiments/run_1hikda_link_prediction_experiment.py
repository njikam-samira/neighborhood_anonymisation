from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.evaluation.run_1hikda_link_prediction_experiment import main


if __name__ == "__main__":
    main(sys.argv[1:])

