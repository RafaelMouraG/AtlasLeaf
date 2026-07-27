"""Entrada canônica para avaliar o ONNX em fotos reais de campo."""

import sys
from pathlib import Path

BACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACK))

from atlasleaf.evaluation import main


if __name__ == "__main__":
    main()
