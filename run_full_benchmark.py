from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.benchmarks.full_benchmark import *  # noqa: F401,F403
from graph_anonymization.benchmarks.full_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
