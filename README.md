# 🌿 AtlasLeaf

> Detector de Saúde em Folhas de Soja usando Inteligência Artificial

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-ff4b4b.svg)](https://streamlit.io)

---

## 🎯 Sobre o Projeto

O **AtlasLeaf** é um sistema de visão computacional que utiliza deep learning para detectar se uma folha de soja está saudável ou doente. O projeto foi desenvolvido para auxiliar o meu dia a dia na fazenda.

### Funcionalidades

- ✅ Detecção de folhas saudáveis vs doentes
- ✅ Interface web intuitiva com Streamlit
- ✅ Modelo otimizado em formato ONNX
- ✅ Alta acurácia (~96%)

---

## 🔍 Classificação

| Classe | Descrição |
|--------|-----------|
| ✅ **Healthy** | Folha saudável |
| 🦠 **Disease** | Folha com doença |

---

## 📦 Instalação

### Pré-requisitos

- Python 3.10+
- pip ou conda

### Passos

1. Clone o repositório:
   - `git clone https://github.com/RafaelMouraG/AtlasLeaf.git`
   - `cd AtlasLeaf`

2. Crie um ambiente virtual:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (Linux/macOS)
   - `.venv\Scripts\activate` (Windows)

3. Instale as dependências:
   - `pip install -r requirements.txt`

---

## 🚀 Uso

### Interface Web

1. `source .venv/bin/activate`
2. `cd Back`
3. `streamlit run app_streamlit.py`

Acesse: **http://localhost:8501**

---

## 🔬 Pipeline de Machine Learning

### Dataset: SoyNet

- **Fonte**: [Mendeley Data](https://data.mendeley.com/datasets/w2r855hpx8/2)
- **Total de Imagens**: 3.720
- **Resolução**: 256x256 pixels
- **Origem**: Campos agrícolas indianos

**Distribuição original (desbalanceada):**

| Classe | Imagens | Proporção |
|--------|---------|-----------|
| Disease | 3.164 | 85% |
| Healthy | 556 | 15% |

### Técnicas de Treinamento

| Técnica | Propósito |
|---------|-----------|
| **WeightedRandomSampler** | Oversampling para balancear classes (6:1 → 1:1) |
| **Data Augmentation** | Flip, Rotation, ColorJitter, Affine |
| **ReduceLROnPlateau** | Ajuste automático de learning rate |
| **Transfer Learning** | Fine-tuning em ResNet18 pré-treinada (ImageNet) |

> **Nota:** O dataset original tem proporção 6:1 (Disease/Healthy). Sem balanceamento, o modelo tende a classificar tudo como "doente". O WeightedRandomSampler resolve isso atribuindo peso maior às amostras da classe minoritária.

---

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| **Classes** | 2 (Healthy, Disease) |
| **Dataset** | 3.720 imagens |
| **Acurácia** | ~96% |
| **Arquitetura** | ResNet18 |

---

## 🗺️ Roadmap

- [x] Treinamento com SoyNet Dataset
- [x] Classificação: Healthy vs Disease
- [x] Interface Streamlit
- [x] Export ONNX
- [ ] Aplicativo móvel
- [ ] Identificação de doenças específicas
- [ ] Detecção de pragas

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 Autor

**Rafael Moura**

- GitHub: [@RafaelMouraG](https://github.com/RafaelMouraG)
