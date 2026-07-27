"""Contrato de checkpoints, ONNX e metadata do AtlasLeaf."""

from __future__ import annotations

from typing import Any

from .labels import FIELD7_CLASS_IDS


def classifier_output_count(checkpoint: dict[str, Any]) -> int:
    """Infere a quantidade de logits sem depender de flags externas."""
    return int(checkpoint["model_state_dict"]["classifier.1.weight"].shape[0])


def output_class_ids(checkpoint: dict[str, Any], num_outputs: int) -> list[int]:
    """Obtém o mapeamento saída→ID, mantendo compatibilidade com legados."""
    ids = checkpoint.get("run", {}).get("output_class_ids")
    if ids is None:
        ids = FIELD7_CLASS_IDS if num_outputs == len(FIELD7_CLASS_IDS) else list(range(num_outputs))
    ids = [int(class_id) for class_id in ids]
    if len(ids) != num_outputs:
        raise ValueError("output_class_ids não corresponde ao número de saídas do checkpoint")
    return ids


def requires_output_mask(num_outputs: int) -> bool:
    """Indica checkpoint que ainda exige mascaramento de logits excedentes."""
    return num_outputs != len(FIELD7_CLASS_IDS)
