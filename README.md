# 🌿 AtlasLeaf

> Diagnóstico de doenças em folhas de soja por visão computacional — com **avaliação honesta** de generalização para campo.

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-ff4b4b.svg)](https://streamlit.io)

---

## 🎯 Sobre

O **AtlasLeaf** classifica doenças foliares de soja a partir de uma foto, para apoiar decisões na
lavoura. O modelo atual (**field7**) cobre **7 classes de campo** com uma EfficientNet-V2-S, e — mais
importante — é avaliado de forma **honesta** quanto a generalizar para imagens de fora do dataset de treino.

### A lição central do projeto

Uma primeira versão atingiu **98,5% na validação** e mesmo assim **falhava em imagens externas**. A
investigação mostrou o porquê: cada doença vinha majoritariamente de **uma única fonte** (câmera/região),
então um split aleatório deixava o modelo **decorar a assinatura da fonte** (câmera, compressão, fundo) em
vez da lesão. Provado com um *probe* linear: dá para prever a **fonte** da foto com **99% de acerto**.

> **Regra do projeto:** avaliar sempre com **split por fonte/câmera** (treinar num domínio, testar em
> outro). Split aleatório infla a métrica e mente sobre o desempenho real em campo.

---

## 🧠 Modelo atual — `field7` (`v2.0`)

**Arquitetura:** EfficientNet-V2-S (ImageNet) · fine-tune completo (`lr=1e-4`, `domain_aug`, Mixup/CutMix, Focal Loss) · 7 saídas nativas · entrada 384×384.

**7 classes de campo** (as com lastro real de campo no ASDID + reforço de 2ª fonte no Mendeley V2):

| Classe | Nome | Científico |
|---|---|---|
| `healthy` | Folha sadia | — |
| `asian_rust` | Ferrugem asiática | *Phakopsora pachyrhizi* |
| `target_spot` | Mancha alvo | *Corynespora cassiicola* |
| `cercospora_blight` | Crestamento de cercospora | *Cercospora kikuchii* |
| `downy_mildew` | Míldio | *Peronospora manshurica* |
| `potassium_deficiency` | Deficiência de potássio | — (nutricional) |
| `frogeye_leaf_spot` | Mancha olho-de-rã | *Cercospora sojina* |

### Resultados (honestos)

Avaliação por **split de câmera** (treina na câmera Canon do ASDID + Mendeley V2, testa na câmera Motorola do ASDID — nunca vista no treino).
Acurácia balanceada no teste em **câmera não vista**: **95,3%** (acurácia geral **95,5%**) · limiar calibrado na validação (`0,565`): **96,7%** de acerto cobrindo **97,9%** dos casos.

| Doença | Recall | Precisão | F1 | Imgs teste (Moto) |
|---|---:|---:|---:|---:|
| Deficiência de potássio | 99,5% | 100,0% | 0,998 | 413 |
| Mancha alvo | 99,1% | 99,4% | 0,992 | 326 |
| Crestamento de cercospora | 96,7% | 99,7% | 0,981 | 301 |
| Ferrugem asiática | 92,0% | 98,8% | 0,953 | 275 |
| Míldio | 90,1% | 98,3% | 0,940 | 191 |
| Mancha olho-de-rã | 90,5% | 96,1% | 0,932 | 570 |
| Folha sadia | 99,5% | 83,0% | 0,905 | 401 |

> **Recall** = das folhas que realmente têm a doença, quantas o modelo pega. **Precisão** = quando ele
> aponta a doença, quantas vezes acerta. Métricas do conjunto de teste em câmera não vista (2.477 imagens
> Motorola). Total de treino + validação: 6.976 imagens (ASDID Canon + Mendeley V2).

> 🌍 **Teste cross-região (EUA → Índia):** segurando o Mendeley V2 inteiro fora do treino (295 fotos de
> outro país/câmera), a ConvNeXt-Tiny congelada atingiu **84,4%** de acurácia balanceada. Ainda assim,
> em uma lavoura nova (outra região/cultivar), o número real pode variar — valide com fotos reais da sua
> lavoura usando `scripts/evaluate_field.py`.

**Descoberta da v2.0:** antes o fine-tune completo piorava por usar LR alto (`5e-4`) em fonte única; com
**LR baixo (`1e-4`) + `domain_aug` + 2ª fonte de campo (Mendeley V2)**, o fine-tune superou o backbone
congelado em **+20 pp** (de ~74,6% para **95,3%**, e *Cercospora* de 42% para **96,7%** de recall).
Recorte/segmentação de folha continua descartado.

---

## ⚙️ Como funciona

```
📷 Foto → redimensiona 384×384 → normaliza (ImageNet) → EfficientNet-V2-S → softmax nas 7 classes → diagnóstico + confiança
```

- **Sem recorte** de folha (testado e descartado).
- **Limiar de confiança** (calibrado exclusivamente na validação: `0,565`): abaixo disso a predição é
  marcada como *baixa confiança* e sugere revisão humana. No holdout Moto, atinge **96,7%** de acerto
  cobrindo **97,9%** dos casos — ótimo para apoio à decisão.
- A saída tem **7 logits nativos** (`masked_output: false`), com `output_class_ids` na metadata mapeando
  cada posição ao ID histórico do manifest.

---

## 📦 Instalação

```bash
git clone https://github.com/RafaelMouraG/AtlasLeaf.git
cd AtlasLeaf
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # streamlit>=1.59 (versões antigas quebram no Python 3.14)
```

---

## 🚀 Uso

### Interface web

```bash
cd Back
../.venv/bin/streamlit run streamlit_app.py
```
Carrega automaticamente o modelo `field7`, mostra as 7 doenças e sinaliza baixa confiança.

### Testar com suas próprias fotos (o número que importa)

Organize `minhas_fotos/{nome_da_doenca}/*.jpg` e rode:
```bash
cd Back
../.venv/bin/python scripts/evaluate_field.py --dir minhas_fotos
```
Reporta acurácia, acurácia balanceada, matriz de confusão e a acurácia **só nas predições confiantes**.

### Treinar (reprodutível, GPU CUDA/ROCm ou Google Colab Pro)

```bash
cd Back
../.venv/bin/python scripts/train_field7.py --data-dir datasets/unified_resized \
    --source-split --split-file splits_camera_mendeley_train.json \
    --field7 --domain-aug --amp --epochs 40 --batch-size 16 --accumulation-steps 3 --lr 1e-4 \
    --target-precision 0.999 --ckpt-out artifacts/field7/model.pth
# exportar o ONNX + metadata a partir do checkpoint:
../.venv/bin/python scripts/export_model.py --ckpt artifacts/field7/model.pth
```

> 💡 Em testes, **sempre** use `--ckpt-out` com um nome próprio para não sobrescrever o modelo bom.
> No Google Colab Pro, use [colab_train_atlasleaf.ipynb](Back/colab_train_atlasleaf.ipynb).

Veja [a arquitetura do projeto](Back/docs/architecture.md) para o contrato de datasets/modelos e os comandos canônicos.

---

## 📁 Estrutura do projeto

```
AtlasLeaf/
├── README.md · requirements.txt · LICENSE
└── Back/
    ├── streamlit_app.py              # Interface web (ponto de entrada)
    ├── atlasleaf/                    # treino, rótulos, modelo, inferência e artefatos
    ├── scripts/                      # comandos de treino, export, avaliação e dados
    ├── data_pipeline/
    │   ├── config.py                 # Configuração de treino
    │   ├── model.py                  # EfficientNet-V2-S, Focal Loss, Mixup/CutMix
    │   ├── inference.py              # Pipeline de inferência (TTA, incerteza)
    │   ├── augmentation.py           # Augmentations de campo
    │   ├── domain_aug.py             # Augmentation de domínio (resolução/JPEG/cor)
    │   ├── leaf_segmentation.py      # Recorte de folha (disponível, mas descartado)
    │   ├── source_split.py           # Split por FONTE (avaliação honesta)
    │   └── camera_split.py           # Split por CÂMERA (proxy campo→campo)
    │
    ├── docs/
    │   ├── protocolo_coleta.md       # Como coletar o dataset do oeste baiano
    │   └── prompt_busca_datasets.md  # Prompt p/ Claude/Gemini acharem datasets complementares
    │
    ├── artifacts/                    # ONNX, checkpoints e metadata
    │   └── field7/                   # Modelo em produção
    └── datasets/ (não versionado)
        ├── unified/          # ASDID + Kaggle + SoyNet unificados
        ├── unified_resized/  # Imagens 768px (treino rápido, sem recorte)
        └── unified_cropped/  # Imagens recortadas (experimento descartado)
```

---

## 🔬 Metodologia

- **Avaliação honesta:** `source_split.py` (segura uma fonte inteira p/ teste) e `camera_split.py`
  (treina numa câmera, testa em outra). Nunca split aleatório para medir generalização.
- **Balanceamento suave:** `WeightedRandomSampler` com peso `1/√(contagem)` — sem empilhar oversampling
  + pesos de loss (o empilhamento colapsava as classes grandes).
- **Domain augmentation:** resolução/JPEG/cor aleatórios para apagar a assinatura de fonte nos pixels.
- **Backbone congelado:** preserva as features transferíveis do ImageNet (generaliza melhor que fine-tune).
- **Early stopping** por **acurácia balanceada** da validação (não pela loss, que engana com pesos).

---

## ⚠️ Limitações (honestas)

- Apenas **7 doenças** têm lastro de campo confiável. As demais (estúdio/fonte única) foram **removidas**
  do produto — não são confiáveis em campo e poluiriam as predições.
- **Cercospora** é a classe mais fraca (~42% recall) — precisa de mais dados.
- O ~75% é **cross-camera**, ainda otimista para uma lavoura nova. Valide com fotos suas.
- O dataset base (ASDID) é campo real, mas de **poucas câmeras e provável região única** — daí o plano
  de coleta abaixo.

---

## 🗺️ Roadmap

- [x] Modelo field7 (7 doenças) com avaliação honesta por câmera
- [x] Interface web + limiar de confiança
- [x] Ferramentas de avaliação com fotos reais (`scripts/evaluate_field.py`)
- [x] Protocolo de coleta do oeste baiano (`docs/protocolo_coleta.md`)
- [x] Integração de dataset novo + split por fazenda (`scripts/integrate_dataset.py`)
- [ ] **Coletar dataset do oeste baiano** (várias fazendas/telefones) para virar *in-domain* na região
- [ ] Buscar datasets públicos complementares de campo (`docs/prompt_busca_datasets.md`)
- [ ] Reforçar cercospora e reavaliar com split por região
- [ ] Expandir para doenças novas do Cerrado (antracnose, mofo branco, ...)

### Histórico

- **v1.0** — binário sadia/doente (SoyNet).
- **v2.0** — 10 doenças (Kaggle, dataset pequeno).
- **v3.0/3.1** — unificação de datasets (ASDID + Kaggle + SoyNet) e a descoberta do viés de fonte.
- **field7** *(atual)* — foco nas 7 classes de campo, avaliação honesta, ~75% cross-camera.

---

## 📚 Datasets e créditos

Crédito, licença e uso de cada fonte em [docs/datasets.md](Back/docs/datasets.md).

O modelo publicado **`field7` (`v2.0`) é treinado com ASDID + Mendeley V2** (split por câmera mantendo
Motorola como holdout exclusivo):

- **ASDID** — Bevers, N., Sikora, E. J., & Hardy, N. B. (2022). *Auburn Soybean Disease Image Dataset*.
  Dryad. https://doi.org/10.5061/dryad.41ns1rnj3 · CC0 1.0
- **Mendeley V2** — Thorwat, M., Magdum, P., Jadhav, S., Sutar, A., & Oswal, R. (2026). *Multi-Class
  Soybean Leaf Disease Dataset: Healthy and Diseased Leaf Images for Machine Learning* (V2). Mendeley
  Data. https://data.mendeley.com/datasets/6fhphxg297/2 · **CC BY 4.0**

Outras fontes (**SoyNet**, um dataset **Kaggle**) existem no dataset unificado por herança das versões
antigas, mas **não são usadas pelo modelo atual** — detalhes e citações no doc.

> A **licença MIT abaixo cobre o código**, não as imagens. Cada dataset mantém a sua própria licença.

---

## 📝 Licença

MIT — veja [LICENSE](LICENSE).

## 👨‍💻 Autor

**Rafael Moura** — [@RafaelMouraG](https://github.com/RafaelMouraG)
