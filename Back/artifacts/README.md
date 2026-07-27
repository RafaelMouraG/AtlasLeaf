# Artefatos de modelo

Arquivos gerados pelo treino ficam fora da raiz de `Back/`.

- `field7/`: modelo em produção. Contém `model.pth`, `model.onnx` e
  `metadata.json`.

O código deve usar os caminhos de `atlasleaf.paths`, nunca nomes de artefatos
espalhados em scripts. Checkpoints e ONNX são ignorados pelo Git; a metadata do
modelo pode ser versionada quando representar uma release publicada.
