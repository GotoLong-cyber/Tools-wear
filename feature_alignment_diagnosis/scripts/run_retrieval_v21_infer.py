#!/usr/bin/env python3
"""Formal inference entry for Retrieval V2.1.

This keeps the original experiment script intact and exposes a stable name for
current best test-time retrieval inference.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_alignment_diagnosis.scripts.evaluate_fold1_knn_delta_retrieval import main


if __name__ == "__main__":
    main()
