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

## 🧠 Modelo atual — `field7`

**Arquitetura:** EfficientNet-V2-S (ImageNet) · backbone congelado, só a cabeça treinada · entrada 384×384.

**7 classes de campo** (as com lastro real de campo no ASDID):

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

Avaliação por **split de câmera** (treina na câmera Canon, testa na Motorola — proxy campo→campo).
Acurácia balanceada no teste em **câmera não vista**: **74,6%** · in-domain (mesma câmera) ~88% *(otimista, não representa deploy)*.

| Doença | Recall | Precisão | F1 | Imgs treino (Canon) | Imgs teste (Moto) |
|---|---:|---:|---:|---:|---:|
| Folha sadia | 91% | 52% | 0,66 | 1297 | 452 |
| Míldio | 87% | 59% | 0,71 | 461 | 191 |
| Deficiência de potássio | 86% | 88% | 0,87 | 619 | 413 |
| Mancha alvo | 77% | 94% | 0,84 | 753 | 326 |
| Ferrugem asiática | 70% | 84% | 0,77 | 1431 | 342 |
| Mancha olho-de-rã | 68% | 85% | 0,76 | 970 | 570 |
| Crestamento de cercospora | 42% | 96% | 0,58 | 1351 | 361 |

> **Recall** = das folhas que realmente têm a doença, quantas o modelo pega. **Precisão** = quando ele
> aponta a doença, quantas vezes acerta. Métricas do conjunto de teste em câmera não vista (2.655 imagens
> Motorola); numa lavoura diferente tendem a ser um pouco menores. Total de treino: 6.882 imagens (ASDID, câmera Canon).

> ⚠️ Em uma lavoura **diferente** (outra região/telefone/cultivar) o número real tende a ficar **abaixo
> de 75%** — Canon e Motorola dividem as mesmas parcelas. O teste que fecha a questão é rodar com fotos
> reais da sua lavoura (ver `scripts/evaluate_field.py`).

**Descartado por não ajudar:** recorte/segmentação de folha (o atalho de fonte vive nos pixels da folha,
não no fundo) e fine-tune completo (distorce as features e generaliza **pior** que o backbone congelado).

---

## ⚙️ Como funciona

```
📷 Foto → redimensiona 384×384 → normaliza (ImageNet) → EfficientNet-V2-S → softmax nas 7 classes → diagnóstico + confiança
```

- **Sem recorte** de folha (testado e descartado).
- **Limiar de confiança** (padrão 0,6): abaixo disso a predição é marcada como *baixa confiança* e
  sugere revisão humana. Em teste, no limiar 0,6 o modelo defere ~parte dos casos mas acerta quase 100%
  nos que responde — bom para uso como apoio à decisão.
- A saída é renormalizada **apenas entre as 7 classes suportadas** (o `.onnx` tem 15 saídas por legado;
  `supported_class_ids` na metadata define quais valem).

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

### Treinar (reprodutível, ~2 min/época no dataset reduzido)

```bash
cd Back
../.venv/bin/python scripts/train_field7.py --data-dir datasets/unified_resized \
    --source-split --split-file splits_camera.json \
    --field7 --freeze-backbone --domain-aug --epochs 40 \
    --ckpt-out artifacts/field7/model.pth
# exportar o ONNX + metadata a partir do checkpoint:
../.venv/bin/python scripts/export_model.py --ckpt artifacts/field7/model.pth
```

> 💡 Em testes, **sempre** use `--ckpt-out` com um nome próprio para não sobrescrever o modelo bom.

> O artefato `field7` antigo possui 15 saídas mascaradas e métricas legadas não rastreáveis. Gere um novo
> checkpoint com `--field7` para obter 7 saídas nativas, limiar calibrado na validação e métricas anexadas ao checkpoint.

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

O modelo publicado **field7 é treinado exclusivamente com o ASDID** (o split por câmera só inclui
imagens dessa fonte):

- **ASDID** — Bevers, N., Sikora, E. J., & Hardy, N. B. (2022). *Auburn Soybean Disease Image Dataset*.
  Dryad. https://doi.org/10.5061/dryad.41ns1rnj3 · CC0 1.0

Outras fontes (**SoyNet**, um dataset **Kaggle**) existem no dataset unificado por herança das versões
antigas, mas **não são usadas pelo modelo atual** — detalhes e citações no doc.

> A **licença MIT abaixo cobre o código**, não as imagens. Cada dataset mantém a sua própria licença.

---

## 📝 Licença

MIT — veja [LICENSE](LICENSE).

## 👨‍💻 Autor

**Rafael Moura** — [@RafaelMouraG](https://github.com/RafaelMouraG)
