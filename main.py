from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.benchmarks.full_benchmark import main as run_full_benchmark_main
from graph_anonymization.benchmarks.hikda_benchmark import main as run_1hikda_benchmark_main
from graph_anonymization.evaluation.run_1hikda_link_prediction_experiment import (
    main as run_1hikda_link_prediction_main,
)
from graph_anonymization.evaluation.run_link_prediction_experiment import (
    main as run_link_prediction_main,
)


HELP_TEXT = """Usage:
  python main.py full-benchmark [args...]
  python main.py link-prediction [args...]
  python main.py hikda-benchmark [args...]
  python main.py hikda-link-prediction [args...]

Commands:
  full-benchmark          Benchmark Ange original / Ange modifie NCC / Zhou-Pei
  link-prediction         Utility evaluation by link prediction
  hikda-benchmark         Benchmark 1HIkDA
  hikda-link-prediction   Link prediction evaluation for 1HIkDA
"""


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP_TEXT)
        return 0

    command = args[0]
    remaining = args[1:]

    if command == "full-benchmark":
        return int(run_full_benchmark_main(remaining))
    if command == "link-prediction":
        run_link_prediction_main(remaining)
        return 0
    if command == "hikda-benchmark":
        return int(run_1hikda_benchmark_main(remaining))
    if command == "hikda-link-prediction":
        run_1hikda_link_prediction_main(remaining)
        return 0

    print(f"Unknown command: {command}\n")
    print(HELP_TEXT)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

