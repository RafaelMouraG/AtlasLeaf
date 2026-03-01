# 🌿 AtlasLeaf

> Sistema de Diagnóstico de Doenças em Folhas de Soja usando Inteligência Artificial

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-ff4b4b.svg)](https://streamlit.io)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2-brightgreen.svg)](https://spring.io/projects/spring-boot)

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Versões do Sistema](#-versões-do-sistema)
- [Como Funciona](#-como-funciona)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [API REST (Java Spring)](#-api-rest-java-spring)
- [Roadmap](#-roadmap)

---

## 🎯 Sobre o Projeto

O **AtlasLeaf** é um sistema de visão computacional que utiliza deep learning para identificar doenças em folhas de soja. O projeto foi desenvolvido para auxiliar o meu dia a dia na fazenda.

### Funcionalidades

- ✅ Detecção de folhas doentes vs saudáveis (v1.0)
- ✅ Identificação de 10 doenças específicas (v2.0)
- ✅ Sistema em cascata para maior precisão
- ✅ Interface web com Streamlit

---

## 🔄 Versões do Sistema

### v1.0 - Classificação Binária
| Classe | Descrição |
|--------|-----------|
| ✅ **Healthy** | Folha saudável |
| 🦠 **Disease** | Folha com doença |

**Dataset:** SoyNet (3.720 imagens)

### v2.0 - Classificação Multi-Classe (10 Doenças)
| Doença | Nome Técnico | Severidade |
|--------|--------------|------------|
| 🔴 Ferrugem Asiática | *Phakopsora pachyrhizi* | Alta |
| 🔴 Síndrome da Morte Súbita | *Fusarium virguliforme* | Alta |
| 🔴 Podridão do Colo | *Sclerotium rolfsii* | Alta |
| 🟠 Mancha Bacteriana | *Pseudomonas syringae* | Média |
| 🟠 Mancha Marrom | *Septoria glycines* | Média |
| 🟠 Septoriose | *Septoria* spp. | Média |
| 🟠 Crestamento | Queima foliar | Média |
| 🟡 Vírus do Mosaico | *SMV* | Baixa |
| 🟡 Mosaico Amarelo | *BYMV* | Baixa |
| 🟡 Oídio | *Microsphaera diffusa* | Baixa |

**Dataset:** Soybean Diseased Leaf Dataset - Kaggle (609 imagens)

### Sistema em Cascata (Recomendado)
Combina v1.0 + v2.0 para maior precisão:

```
📸 Imagem
    ↓
┌─────────────────────────┐
│ ETAPA 1: v1.0           │
│ Está doente? (3.720 img)│
└─────────────────────────┘
    ↓
  Saudável? → ✅ "Folha Saudável"
    ↓
  Doente?
    ↓
┌─────────────────────────┐
│ ETAPA 2: v2.0           │
│ Qual doença? (609 img)  │
└─────────────────────────┘
    ↓
🦠 Diagnóstico específico
```

---

## 🧠 Como Funciona

### Fluxo Simplificado

```
📷 Imagem → 🔄 Pré-processamento → 🧠 Modelo IA → 📊 Diagnóstico
```

### Pré-processamento

```python
# Transformações aplicadas:
1. Resize → 256x256 pixels
2. Conversão → RGB (3 canais)
3. Normalização → valores entre 0-1
4. Padronização ImageNet → mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
5. Formato → Tensor [1, 3, 256, 256]
```

### Inferência

```python
# Exemplo de saída v1.0:
[0.85, 0.15]  # [Disease: 85%, Healthy: 15%]

# Exemplo de saída v2.0:
[0.02, 0.01, 0.05, 0.03, 0.02, 0.01, 0.75, 0.08, 0.02, 0.01]  # Ferrugem: 75%
```

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ATLASLEAF - ARQUITETURA                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │   Frontend   │    │     Backend      │    │      Modelos       │    │
│  ├──────────────┤    ├──────────────────┤    ├────────────────────┤    │
│  │  Streamlit   │───▶│  Python/Spring   │───▶│   ONNX Runtime     │    │
│  │              │    │                  │    │                    │    │
│  │  • Upload    │    │  • Cascade Logic │    │  • v1.0 (2 cls)    │    │
│  │  • Display   │    │  • Preprocessing │    │  • v2.0 (10 cls)   │    │
│  │  • Ranking   │    │  • REST API      │    │  • ResNet18        │    │
│  └──────────────┘    └──────────────────┘    └────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tecnologias

| Componente | Tecnologia | Versão |
|------------|------------|--------|
| **Treinamento** | PyTorch | 2.9+ |
| **Modelo** | ResNet18 | ImageNet pretrained |
| **Inferência** | ONNX Runtime | 1.16+ |
| **Interface Web** | Streamlit | 1.29+ |
| **API REST** | Spring Boot | 3.2 |

---

## 📦 Instalação

### Pré-requisitos

- Python 3.10+
- pip ou conda
- (Opcional) Java 17+ para API Spring

### Instalação Python

```bash
# 1. Clone o repositório
git clone https://github.com/RafaelMouraG/AtlasLeaf.git
cd AtlasLeaf

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. (Opcional) Para treinar os modelos:
pip install torch torchvision onnx onnxruntime onnxscript
```

---

## 🚀 Uso

### Interface Web - Sistema em Cascata (Recomendado)

```bash
source .venv/bin/activate
cd Back
streamlit run app_streamlit_cascade.py
```
Acesse: **http://localhost:8501**

### Interface Web - v1.0 (Disease/Healthy)

```bash
streamlit run app_streamlit.py
```

### Interface Web - v2.0 (10 Doenças)

```bash
streamlit run app_streamlit_v2.py
```

### Treinamento dos Modelos

```bash
cd Back

# v1.0 - Classificação Binária
# Abra train_atlasleaf.py no VS Code como notebook

# v2.0 - Multi-Classe (10 doenças)
# Abra train_atlasleaf_v2.ipynb no VS Code/Jupyter
```

---

## 📁 Estrutura do Projeto

```
AtlasLeaf/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 LICENSE
│
└── Back/
    │
    ├── 🌐 Interfaces Web
    │   ├── app_streamlit.py          # v1.0 (Disease/Healthy)
    │   ├── app_streamlit_v2.py       # v2.0 (10 doenças)
    │   └── app_streamlit_cascade.py  # Sistema em cascata ⭐
    │
    ├── 📚 Treinamento
    │   ├── train_atlasleaf.py        # Notebook v1.0
    │   ├── train_atlasleaf_v2.ipynb  # Notebook v2.0
    │   └── train_atlasleaf_v3.ipynb  # Notebook v3.0 (Big Data) ⭐
    │
    ├── 🔧 Data Pipeline (v3.0)
    │   └── data_pipeline/
    │       ├── __init__.py
    │       ├── config.py             # Taxonomia unificada (17 classes)
    │       ├── dataset_unifier.py    # Unifica múltiplos datasets
    │       ├── losses.py             # Focal Loss, Class-Balanced
    │       └── augmentation.py       # Augmentation adaptativa
    │
    ├── 🧠 Modelos v1.0
    │   ├── atlasleaf_soybean.onnx
    │   ├── atlasleaf_metadata.json
    │   └── atlasleaf_best_model.pth
    │
    ├── 🧠 Modelos v2.0
    │   ├── atlasleaf_v2_diseases.onnx
    │   ├── atlasleaf_v2_metadata.json
    │   └── atlasleaf_v2_best_model.pth
    │
    ├── 📊 Datasets (não versionados)
    │   ├── SoyNet_Dataset/           # v1.0 (3.720 imgs)
    │   ├── Soybean_Diseases_Dataset/ # v2.0 Kaggle (609 imgs)
    │   └── datasets/                 # v3.0 Big Data
    │       ├── raw/                  # Datasets originais
    │       │   ├── ASDID/            # Auburn (~40GB)
    │       │   ├── Digipathos/       # Embrapa
    │       │   └── INSECT12C/        # Pragas
    │       └── unified/              # Dataset unificado
    │
    └── atlasleaf-api/                # API Spring Boot 🚧
        ├── pom.xml
        └── src/
            └── main/
                ├── java/com/atlasleaf/
                │   ├── config/
                │   ├── controller/
                │   ├── service/
                │   ├── model/
                │   │   ├── dto/
                │   │   └── domain/
                │   └── exception/
                └── resources/
                    ├── application.yml
                    └── models/
```

---

## 🔬 Pipeline de Machine Learning

### Estratégia de Dados v3.0

> ⚠️ **Problema identificado**: O dataset Kaggle (609 imgs) é insuficiente para generalização em campo.

#### Datasets Recomendados (Big Data)

| Dataset | Imagens | Fonte | Uso |
|---------|---------|-------|-----|
| **ASDID** (Auburn) | ~10.000+ | [Zenodo](https://zenodo.org) | Treino principal |
| **SoyNet** | ~9.300 | [Mendeley](https://data.mendeley.com) | Validação mobile |
| **Digipathos** (Embrapa) | ~372+ | [Embrapa](https://www.digipathos.cnptia.embrapa.br/) | Gold standard |
| **INSECT12C** (UFGD) | ~6.000 | GitHub | Exclusão de pragas |
| Kaggle (atual) | 609 | Kaggle | Legado |

#### Pipeline de Unificação

```bash
# Após baixar os datasets, execute:
cd Back
python data_pipeline/dataset_unifier.py --datasets asdid soynet kaggle_soybean

# Isso criará:
# - datasets/unified/        → Imagens organizadas por classe
# - datasets/unified/manifest.json → Metadados
# - datasets/unified/splits.json   → Train/Val/Test
```

#### Técnicas para Desequilíbrio de Classes

| Técnica | Implementação | Arquivo |
|---------|---------------|---------|
| **Focal Loss** | γ=2.0, reduz peso de exemplos fáceis | `data_pipeline/losses.py` |
| **Class-Balanced Loss** | Pesos pelo número efetivo de amostras | `data_pipeline/losses.py` |
| **Weighted Sampler** | Oversampling de classes minoritárias | `train_atlasleaf_v3.ipynb` |
| **Augmentation Adaptativa** | Mais agressiva para classes raras | `data_pipeline/augmentation.py` |
| **Mixup/CutMix** | Regularização por interpolação | `data_pipeline/losses.py` |

### Dataset v1.0: SoyNet

- **Fonte**: [Mendeley Data](https://data.mendeley.com/datasets/w2r855hpx8/2)
- **Imagens**: 3.720 (Disease + Healthy)
- **Resolução**: 256x256 pixels

### Dataset v2.0: Soybean Diseased Leaf

- **Fonte**: Kaggle
- **Imagens**: 609 (10 classes de doenças)
- **Desafio**: Dataset desbalanceado (5-137 imgs/classe)

### Modelo: ResNet18 (Transfer Learning)

```
ResNet18 (ImageNet)
    │
    ├──▶ v1.0: fc → 2 classes
    │
    └──▶ v2.0: layer3 + layer4 + fc → 10 classes
              (fine-tuning mais profundo)
```

### Treinamento v2.0 - Técnicas Especiais

| Técnica | Propósito |
|---------|-----------|
| **WeightedRandomSampler** | Balanceamento de classes |
| **Class-weighted Loss** | Penaliza erros em classes raras |
| **Heavy Augmentation** | RandomCrop, Perspective, GaussianBlur, Erasing |
| **OneCycleLR** | Scheduler com warmup |
| **Early Stopping** | Patience=10 épocas |
| **Gradient Clipping** | Estabilidade do treino |

---

## ☕ API REST (Java Spring)

### Endpoints (em desenvolvimento)

```
POST /api/v1/diagnostic           # Sistema cascata completo
POST /api/v1/diagnostic/detect    # Só v1.0 (disease/healthy)
POST /api/v1/diagnostic/identify  # Só v2.0 (10 doenças)
GET  /api/v1/diseases             # Lista doenças detectáveis
GET  /api/v1/health               # Healthcheck
```

### Tecnologias

- **Spring Boot 3.2** - Framework
- **ONNX Runtime Java** - Inferência
- **SpringDoc OpenAPI** - Swagger automático
- **Lombok** - Redução de boilerplate

### Executar

```bash
cd Back/atlasleaf-api
mvn spring-boot:run
```

Swagger UI: **http://localhost:8080/api/swagger-ui.html**

---

## 🗺️ Roadmap

### ✅ v1.0 - Classificação Binária
- [x] Treinamento com SoyNet Dataset
- [x] Classificação: Healthy vs Disease
- [x] Interface Streamlit
- [x] Export ONNX

### ✅ v2.0 - Classificação Multi-Classe
- [x] 10 doenças específicas
- [x] Dataset Kaggle integrado
- [x] Técnicas para dataset pequeno
- [x] Sistema em cascata (v1+v2)

### � v3.0 - Big Data & Precisão (Em Desenvolvimento)
- [x] Pipeline de unificação de datasets
- [x] Focal Loss para desequilíbrio de classes
- [x] Data Augmentation adaptativa
- [ ] Integração ASDID (Auburn - ~10k imgs)
- [ ] Integração Embrapa Digipathos
- [ ] 15+ classes de doenças
- [ ] Detecção de pragas (INSECT12C)

### 🔮 v4.0 - Mobile & Expansão (Futuro)
- [ ] Aplicativo móvel (Flutter/React Native)
- [ ] Imagens multiespectrais
- [ ] Integração com drones (UAV)
- [ ] Geolocalização de focos

---

## �� Resultados

| Modelo | Classes | Dataset | Acurácia | Status |
|--------|---------|---------|----------|--------|
| v1.0 | 2 | 3.720 imgs | ~85% | ✅ Produção |
| v2.0 | 10 | 609 imgs | ~65%* | ✅ Produção |
| Cascata | 2→10 | Combinado | Melhor precisão | ✅ Produção |
| **v3.0** | 15 | 10k+ imgs | Em treinamento | 🚧 Dev |

*v2.0 limitado pelo tamanho do dataset - v3.0 resolverá isso

### Distribuição de Classes (Problema Atual)

| Classe | v2.0 (Kaggle) | Status |
|--------|---------------|--------|
| Oídio | 137 | 🟡 |
| Morte Súbita | 110 | 🟡 |
| Mosaico Amarelo | 110 | 🟡 |
| Ferrugem | 65 | 🔴 Crítico |
| Podridão Colo | 62 | 🔴 |
| M. Bacteriana | 50 | 🔴 |
| M. Marrom | 27 | 🔴 |
| V. Mosaico | 22 | 🔴 |
| Septoriose | 21 | 🔴 |
| **Crestamento** | **5** | 🔴 **Gravíssimo** |

> A v3.0 com ASDID terá ~1000+ imagens de Ferrugem Asiática!

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 Autor

**Rafael Moura**

- GitHub: [@RafaelMouraG](https://github.com/RafaelMouraG)
