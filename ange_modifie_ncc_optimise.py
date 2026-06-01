from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Optimized public entrypoint, kept as a separate file for clarity in reports.
from graph_anonymization.anonymization.ange_modified import *  # noqa: F401,F403
