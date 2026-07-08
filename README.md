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

Avaliação por **split de câmera** (treina Canon, testa Motorola — proxy campo→campo):

| Métrica | Valor |
|---|---|
| Acurácia balanceada (câmera não vista) | **~75%** |
| In-domain (mesma câmera) | ~88–98% *(otimista, não representa deploy)* |

Recall por classe (câmera não vista): potássio 86% · míldio 87% · sadia 91% · mancha alvo 77% ·
ferrugem 70% · olho-de-rã 68% · **crestamento de cercospora 42%** (classe mais fraca).

> ⚠️ Em uma lavoura **diferente** (outra região/telefone/cultivar) o número real tende a ficar **abaixo
> de 75%** — Canon e Motorola dividem as mesmas parcelas. O teste que fecha a questão é rodar com fotos
> reais da sua lavoura (ver `test_field.py`).

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
../.venv/bin/streamlit run app_streamlit_v31.py
```
Carrega automaticamente o modelo `field7`, mostra as 7 doenças e sinaliza baixa confiança.

### Testar com suas próprias fotos (o número que importa)

Organize `minhas_fotos/{nome_da_doenca}/*.jpg` e rode:
```bash
cd Back
../.venv/bin/python test_field.py --dir minhas_fotos
```
Reporta acurácia, acurácia balanceada, matriz de confusão e a acurácia **só nas predições confiantes**.

### Treinar (reprodutível, ~2 min/época no dataset reduzido)

```bash
cd Back
../.venv/bin/python train_atlasleaf_v31.py --data-dir datasets/unified_resized \
    --source-split --split-file splits_camera.json \
    --freeze-backbone --domain-aug --epochs 40 --ckpt-out atlasleaf_field7_best.pth
# exportar o ONNX + metadata a partir do checkpoint:
../.venv/bin/python export_field7.py --ckpt atlasleaf_field7_best.pth
```

> 💡 Em testes, **sempre** use `--ckpt-out` com um nome próprio para não sobrescrever o modelo bom.

---

## 📁 Estrutura do projeto

```
AtlasLeaf/
├── README.md · requirements.txt · LICENSE
└── Back/
    ├── app_streamlit_v31.py          # Interface web (modelo field7)
    ├── train_atlasleaf_v31.py        # Treino c/ split por fonte/câmera, freeze, domain-aug
    ├── export_field7.py              # Exporta ONNX + metadata das 7 classes
    ├── test_field.py                 # Avaliação com fotos reais do usuário
    ├── integrate_bahia.py            # Junta dataset novo (Bahia) ao ASDID + split por fazenda
    ├── preprocess_leaf_crop.py       # Pré-processa dataset (--no-crop reduz resolução p/ treino rápido)
    │
    ├── data_pipeline/
    │   ├── config_v31.py             # Configuração de treino
    │   ├── model_v31.py              # EfficientNet-V2-S, Focal Loss, Mixup/CutMix
    │   ├── inference_v31.py          # Pipeline de inferência (TTA, incerteza)
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
    ├── atlasleaf_field7_*.{onnx,json,pth}   # Modelo atual + metadata
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
- [x] Ferramentas de avaliação com fotos reais (`test_field.py`)
- [x] Protocolo de coleta do oeste baiano (`docs/protocolo_coleta.md`)
- [x] Integração de dataset novo + split por fazenda (`integrate_bahia.py`)
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

## 📝 Licença

MIT — veja [LICENSE](LICENSE).

## 👨‍💻 Autor

**Rafael Moura** — [@RafaelMouraG](https://github.com/RafaelMouraG)
