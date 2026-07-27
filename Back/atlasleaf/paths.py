"""Caminhos canônicos do projeto, independentes do diretório de execução."""

from pathlib import Path


BACK_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BACK_DIR / "artifacts"
FIELD7_ARTIFACT_DIR = ARTIFACTS_DIR / "field7"


def from_back(path: str | Path) -> Path:
    """Resolve caminhos relativos a `Back/`, mantendo suporte a caminhos absolutos."""
    path = Path(path)
    return path if path.is_absolute() else BACK_DIR / path
