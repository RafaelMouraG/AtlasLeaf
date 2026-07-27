"""Entrada canônica para integrar uma nova coleta ao dataset unificado."""

import sys
from pathlib import Path

BACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACK))

from atlasleaf.integration import main


if __name__ == "__main__":
    main()
