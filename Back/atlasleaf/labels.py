"""Contrato de rótulos do produto field7.

Os IDs são os IDs históricos do manifest. A saída do classificador field7 usa
índices contíguos e ``output_class_ids`` faz a ponte entre ambos.
"""

FIELD7_CLASS_IDS = [0, 1, 2, 3, 7, 12, 13]

_FIELD7_CLASSES = [
    {"id": "0", "name": "healthy", "friendly_name": "Saudável", "scientific_name": "N/A", "severity": "none"},
    {"id": "1", "name": "asian_rust", "friendly_name": "Ferrugem Asiática", "scientific_name": "Phakopsora pachyrhizi", "severity": "critical"},
    {"id": "2", "name": "target_spot", "friendly_name": "Mancha Alvo", "scientific_name": "Corynespora cassiicola", "severity": "high"},
    {"id": "3", "name": "cercospora_blight", "friendly_name": "Crestamento de Cercospora", "scientific_name": "Cercospora kikuchii", "severity": "high"},
    {"id": "7", "name": "downy_mildew", "friendly_name": "Míldio", "scientific_name": "Peronospora manshurica", "severity": "medium"},
    {"id": "12", "name": "potassium_deficiency", "friendly_name": "Deficiência de Potássio", "scientific_name": "N/A", "severity": "medium"},
    {"id": "13", "name": "frogeye_leaf_spot", "friendly_name": "Mancha Olho-de-Rã", "scientific_name": "Cercospora sojina", "severity": "medium"},
]


def field7_classes() -> list[dict]:
    """Retorna cópias, seguras para serialização e alteração pelo chamador."""
    return [dict(item) for item in _FIELD7_CLASSES]


def field7_label_mapping() -> dict[int, int]:
    """Mapeia ID original do dataset para posição da saída do classificador."""
    return {class_id: index for index, class_id in enumerate(FIELD7_CLASS_IDS)}
