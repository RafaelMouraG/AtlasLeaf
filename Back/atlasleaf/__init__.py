"""Núcleo reutilizável do AtlasLeaf.

Os módulos neste pacote são a API estável do projeto. Os scripts em
``scripts/`` são a única superfície de linha de comando suportada.
"""

from .labels import FIELD7_CLASS_IDS, field7_classes
from .paths import ARTIFACTS_DIR, BACK_DIR, FIELD7_ARTIFACT_DIR

__all__ = ["ARTIFACTS_DIR", "BACK_DIR", "FIELD7_ARTIFACT_DIR", "FIELD7_CLASS_IDS", "field7_classes"]
