# 🌿 AtlasLeaf

> Sistema de Diagnóstico de Doenças em Folhas de Soja usando Inteligência Artificial

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-ff4b4b.svg)](https://streamlit.io)

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Como Funciona](#-como-funciona)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [Roadmap](#-roadmap)

---

## 🎯 Sobre o Projeto

O **AtlasLeaf** é um sistema de visão computacional que utiliza deep learning para identificar se folhas de soja estão **saudáveis** ou **doentes**. O projeto foi desenvolvido para auxiliar o meu dia a dia na fazenda.

### Versão Atual: v1.0 (Binary Classification)

| Classe | Descrição |
|--------|-----------|
| ✅ **Healthy** | Folha saudável, sem sinais de doença |
| 🦠 **Disease** | Folha com sinais de doença (ferrugem, manchas, etc.) |

---

## 🧠 Como Funciona

### Fluxo Simplificado

```
📷 Imagem → 🔄 Pré-processamento → 🧠 Modelo IA → 📊 Diagnóstico
```

### Explicação Detalhada

#### 1️⃣ **Captura da Imagem**
O usuário envia uma foto de uma folha de soja através da interface web (Streamlit).

#### 2️⃣ **Pré-processamento**
A imagem passa por transformações para ficar no formato esperado pelo modelo:

```python
# Transformações aplicadas:
1. Resize → 256x256 pixels
2. Conversão → RGB (3 canais de cor)
3. Normalização → valores entre 0-1
4. Padronização ImageNet → mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
5. Formato → Tensor [1, 3, 256, 256] (batch, canais, altura, largura)
```

#### 3️⃣ **Inferência (Predição)**
O tensor da imagem é passado pelo modelo neural (ResNet18) que retorna probabilidades para cada classe.

```python
# Exemplo de saída:
[0.15, 0.85]  # [Disease: 15%, Healthy: 85%]
```

#### 4️⃣ **Resultado**
O sistema mostra a classe com maior probabilidade e o nível de confiança.

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         ATLASLEAF v1.0                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Frontend   │    │   Backend    │    │     Modelo       │  │
│  │  (Streamlit) │───▶│  (Python)    │───▶│  (ONNX Runtime)  │  │
│  │              │    │              │    │                  │  │
│  │  • Upload    │    │  • Preproc.  │    │  • ResNet18      │  │
│  │  • Display   │    │  • Validação │    │  • 2 classes     │  │
│  │  • Resultado │    │  • API       │    │  • ~11M params   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tecnologias Utilizadas

| Componente | Tecnologia | Propósito |
|------------|------------|-----------|
| **Treinamento** | PyTorch | Framework de Deep Learning |
| **Modelo** | ResNet18 | Rede Neural Convolucional pré-treinada |
| **Inferência** | ONNX Runtime | Execução otimizada do modelo |
| **Interface** | Streamlit | Interface web interativa |
| **Dataset** | SoyNet | 3.720 imagens reais de campo |

---

## 📦 Instalação

### Pré-requisitos

- Python 3.10+
- pip ou conda

### Passos

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

# 4. (Opcional) Para treinar o modelo, instale também:
pip install torch torchvision onnx onnxruntime onnxscript
```

---

## 🚀 Uso

### Interface Web (Streamlit)

```bash
cd Back
# 1. Ativar o ambiente virtual
source .venv/bin/activate

# 2. Instalar o Streamlit
pip install streamlit

# 3. Depois pode rodar
cd Back
streamlit run app_streamlit.py
```

Acesse: http://localhost:8501

### Treinamento do Modelo

1. Baixe o dataset SoyNet e extraia em `Back/SoyNet_Dataset/`
2. Execute o notebook de treinamento:

```bash
cd Back
# Abra train_atlasleaf.py no VS Code como notebook
# Execute todas as células
```

---

## 📁 Estrutura do Projeto

```
AtlasLeaf/
├── 📄 README.md                    # Este arquivo
├── 📄 requirements.txt             # Dependências Python
├── 📄 LICENSE                      # Licença MIT
│
└── Back/                           # Backend e ML
    ├── 📓 train_atlasleaf.py       # Notebook de treinamento
    ├── 🌐 app_streamlit.py         # Interface web
    ├── 🧠 atlasleaf_soybean.onnx   # Modelo treinado (ONNX)
    ├── 📊 atlasleaf_metadata.json  # Metadados do modelo
    ├── 💾 atlasleaf_best_model.pth # Checkpoint PyTorch
    │
    └── SoyNet_Dataset/             # Dataset (não versionado)
        └── SoyNet/
            ├── Preprocessing_SoyNet_Data/
            │   ├── Camera Clicks_256_256/
            │   │   ├── Disease_Preprocessing data/
            │   │   └── Healthy_Preprocessing data/
            │   └── ...
            └── Raw_SoyNet_Data/
```

---

## 🔬 Pipeline de Machine Learning

### 1. Dataset: SoyNet

- **Fonte**: [Mendeley Data](https://data.mendeley.com/datasets/w2r855hpx8/2)
- **Imagens**: 3.720 (3.164 Disease + 556 Healthy)
- **Origem**: Campos agrícolas da Índia
- **Resolução**: 256x256 pixels

### 2. Modelo: ResNet18 (Transfer Learning)

```
ResNet18 (ImageNet) ──▶ Fine-tuning ──▶ AtlasLeaf Model
      │                      │                │
 11M parâmetros      Camadas finais      2 classes
                      retreinadas       (Disease/Healthy)
```

**Por que Transfer Learning?**
- Aproveita conhecimento de 1.2M imagens do ImageNet
- Reduz tempo de treinamento drasticamente
- Funciona bem com datasets menores (~3k imagens)

### 3. Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Épocas | 15 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Data Augmentation | Flip, Rotation, ColorJitter |

### 4. Export para ONNX

O modelo PyTorch é convertido para **ONNX** (Open Neural Network Exchange) para:
- ✅ Maior velocidade de inferência
- ✅ Independência de framework
- ✅ Compatibilidade com Java (Spring Boot), C++, JavaScript, etc.

---

## 🗺️ Roadmap

### ✅ v1.0 - Classificação Binária (Atual)
- [x] Treinamento com SoyNet Dataset
- [x] Classificação: Healthy vs Disease
- [x] Interface Streamlit
- [x] Export ONNX

### 🔜 v2.0 - Classificação Multi-Classe
- [ ] Identificar doenças específicas:
  - Ferrugem Asiática
  - Mancha Bacteriana
  - Míldio
  - Mosaico
  - etc.
- [ ] Integrar dataset de doenças específicas

### 🔮 v3.0 - API REST + Mobile
- [ ] API Spring Boot (Java)
- [ ] Aplicativo móvel
- [ ] Detecção de pragas (lagartas, etc.)

---

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| **Acurácia de Validação** | ~85%+ |
| **Classes** | 2 (Disease, Healthy) |
| **Tempo de Inferência** | <100ms |

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer fork do projeto
2. Criar uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abrir um Pull Request

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 Autor

**Rafael Moura**

- GitHub: [@RafaelMouraG](https://github.com/RafaelMouraG)

---

<div align="center">
  <p>🌿 <strong>AtlasLeaf</strong> - IA para agricultura sustentável</p>
  <p>Feito com ❤️ para o agronegócio brasileiro</p>
</div>