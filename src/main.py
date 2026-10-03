"""Run the data preparation pipeline.

Supports both:

    python -m src.main
    python src/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.download_data import download_data
    from src.data.feature_engineering import create_features
    from src.data.preprocess import preprocess_data
else:
    from .data.download_data import download_data
    from .data.feature_engineering import create_features
    from .data.preprocess import preprocess_data


def main() -> None:
    """Run the baseline data preparation pipeline."""
    download_data()
    preprocess_data()
    create_features()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
