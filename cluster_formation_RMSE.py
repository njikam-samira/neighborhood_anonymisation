from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_anonymization.anonymization.cluster_formation_rmse import (  # noqa: F401
    cluster_formation_RMSE,
)

# Backward compatibility alias used by legacy scripts.
cluster_formation = cluster_formation_RMSE

__all__ = ["cluster_formation_RMSE", "cluster_formation"]
