"""Entrada canônica para reduzir ou recortar o dataset."""

import sys
from pathlib import Path

BACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACK))

from atlasleaf.preprocessing import main


if __name__ == "__main__":
    main()
