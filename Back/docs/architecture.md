# Arquitetura do AtlasLeaf

## Contratos estáveis

O produto gira em torno de dois contratos simples:

1. **Dataset**: `manifest.json` descreve cada imagem e `splits_*.json` define
   treino, validação e holdout. Um split é imutável para uma execução e seu hash
   é salvo no checkpoint.
2. **Artefato de modelo**: um diretório contém `model.pth` (treino),
   `model.onnx` (inferência) e `metadata.json` (classes, preprocessamento,
   calibração, métricas e rastreabilidade).

`metadata.json` inclui `output_class_ids`: a posição de cada logit é mapeada ao
ID histórico do manifest. O field7 novo tem exatamente sete logits e não usa
mascaramento de classes legadas.

## Pastas

- `atlasleaf/`: domínio do produto: treino, rótulos, export, inferência e dados.
- `scripts/`: CLIs pequenos e documentados.
- `data_pipeline/`: transformações de imagem e estratégias de split reutilizáveis.

## Ciclo de uma versão

```text
dataset + split imutável
        ↓
scripts/train_field7.py → checkpoint com config, hash e calibração
        ↓
scripts/export_model.py → ONNX + metadata
        ↓
streamlit_app.py / scripts/evaluate_field.py
```

Ao comparar receitas, trate Motorola (ou outra fazenda/região) como holdout:
não escolha hiperparâmetros nem limiar com esse conjunto.
