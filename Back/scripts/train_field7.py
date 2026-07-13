"""Entrada canônica para treinar field7.

Mantém os argumentos do treinador existente e sempre ativa ``--field7``.
"""

import sys
from pathlib import Path

BACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACK))

from atlasleaf.training import main


if __name__ == "__main__":
    if "--field7" not in sys.argv:
        sys.argv.append("--field7")
    main()
