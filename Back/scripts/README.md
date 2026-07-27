# Scripts do AtlasLeaf

Estes são os comandos canônicos do projeto. Os arquivos Python históricos na
raiz de `Back/` continuam existindo por compatibilidade, mas novos comandos e
documentação devem apontar para esta pasta.

- `train_field7.py`: treina o classificador de sete saídas nativas.
- `export_model.py`: exporta checkpoint para ONNX e metadata.
- `evaluate_field.py`: mede desempenho em fotos reais rotuladas.
- `integrate_dataset.py`: incorpora uma nova coleta ao dataset unificado.
- `preprocess_dataset.py`: reduz resolução ou faz recorte experimental.
