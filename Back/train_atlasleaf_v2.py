# %% [markdown]
# # 🌿 AtlasLeaf v2.0 - Classificação Multi-Classe
# 
# Identificação de **10 doenças específicas** em folhas de soja
# 
# Dataset: Soybean Diseased Leaf Dataset (Kaggle)

# %%
"""
AtlasLeaf v2.0 - Treinamento Multi-Classe
Identifica 10 tipos específicos de doenças em soja
"""

# ==================== IMPORTS ====================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
from collections import Counter

# ==================== CONFIGURAÇÕES ====================
CONFIG = {
    'img_size': 256,
    'batch_size': 16,  # Menor batch devido ao dataset menor
    'num_epochs': 30,  # Mais épocas para dataset menor
    'learning_rate': 0.0005,  # LR menor para fine-tuning
    'train_split': 0.8,
    'num_workers': 0,
    'model_name': 'resnet18',
    'output_onnx': 'atlasleaf_v2_diseases.onnx',
    'use_pretrained': True
}

print("🌿 AtlasLeaf v2.0 - Classificação Multi-Classe de Doenças")
print("=" * 65)

# %%
# ==================== 1. CONFIGURAÇÃO DO DATASET ====================
print("\n📂 Configurando Dataset de Doenças...")

BASE_DIR = Path("/Users/rafael/Documents/Personal/Projects/AtlasLeaf/Back")
DISEASES_DATASET = BASE_DIR / "Soybean_Diseases_Dataset"

# Verificar se existe
if not DISEASES_DATASET.exists():
    raise FileNotFoundError(f"❌ Dataset não encontrado em: {DISEASES_DATASET}")

# Listar classes e contar imagens
print(f"\n📊 Classes disponíveis:")
class_counts = {}
for folder in sorted(DISEASES_DATASET.iterdir()):
    if folder.is_dir():
        count = len(list(folder.glob('*.[jJ][pP][gG]')) + list(folder.glob('*.[pP][nN][gG]')))
        class_counts[folder.name] = count
        status = "⚠️" if count < 20 else "✅"
        print(f"   {status} {folder.name}: {count} imagens")

total_images = sum(class_counts.values())
num_classes = len(class_counts)
print(f"\n   Total: {total_images} imagens em {num_classes} classes")

# %%
# ==================== 2. TRANSFORMAÇÕES COM DATA AUGMENTATION PESADO ====================
print("\n🔄 Configurando Data Augmentation (pesado para dataset pequeno)...")

preprocessing_config = {
    'resize': CONFIG['img_size'],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# Augmentation mais agressivo para compensar dataset pequeno
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG['img_size'] + 32, CONFIG['img_size'] + 32)),
    transforms.RandomCrop(CONFIG['img_size']),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=preprocessing_config['mean'], 
                        std=preprocessing_config['std']),
    transforms.RandomErasing(p=0.2)  # Cutout
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=preprocessing_config['mean'], 
                        std=preprocessing_config['std'])
])

# %%
# ==================== 3. CARREGAMENTO DO DATASET ====================
print("\n📚 Carregando Dataset...")

# Carregar com transforms de treino primeiro
full_dataset = datasets.ImageFolder(root=DISEASES_DATASET, transform=train_transforms)

if len(full_dataset) == 0:
    raise ValueError("Dataset vazio!")

# Split
train_size = int(CONFIG['train_split'] * len(full_dataset))
val_size = len(full_dataset) - train_size

# Usar generator para reproducibilidade
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# Calcular pesos para balanceamento de classes
train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts_train = Counter(train_labels)
class_weights = {cls: 1.0 / count for cls, count in class_counts_train.items()}
sample_weights = [class_weights[label] for label in train_labels]

# Sampler balanceado
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'],
    sampler=sampler,  # Usar sampler ao invés de shuffle
    num_workers=CONFIG['num_workers']
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'],
    shuffle=False, 
    num_workers=CONFIG['num_workers']
)

class_names = full_dataset.classes
num_classes = len(class_names)

print(f"✅ Dataset carregado:")
print(f"   - Total: {len(full_dataset)} imagens")
print(f"   - Treino: {len(train_dataset)} | Validação: {len(val_dataset)}")
print(f"   - Classes ({num_classes}): {class_names}")

# %%
# ==================== 4. METADADOS ====================
# Criar nomes mais amigáveis para as classes
friendly_names = {
    'Mossaic Virus': 'Vírus do Mosaico',
    'Southern blight': 'Podridão do Colo',
    'Sudden Death Syndrone': 'Síndrome da Morte Súbita',
    'Yellow Mosaic': 'Mosaico Amarelo',
    'bacterial_blight': 'Mancha Bacteriana',
    'brown_spot': 'Mancha Marrom',
    'crestamento': 'Crestamento',
    'ferrugen': 'Ferrugem Asiática',
    'powdery_mildew': 'Oídio',
    'septoria': 'Septoriose'
}

metadata = {
    'project': 'AtlasLeaf',
    'version': '2.0',
    'task': 'multi-class disease classification',
    'dataset': 'Soybean Diseased Leaf Dataset (Kaggle)',
    'classes': class_names,
    'friendly_names': {cls: friendly_names.get(cls, cls) for cls in class_names},
    'num_classes': num_classes,
    'total_images': len(full_dataset),
    'preprocessing': preprocessing_config,
    'input_shape': [1, 3, CONFIG['img_size'], CONFIG['img_size']],
    'input_name': 'input_image',
    'output_name': 'disease_probabilities',
    'model_architecture': CONFIG['model_name']
}

metadata_file = BASE_DIR / 'atlasleaf_v2_metadata.json'
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✅ Metadados salvos: {metadata_file}")

# %%
# ==================== 5. MODELO ====================
print(f"\n🧠 Inicializando {CONFIG['model_name'].upper()} para {num_classes} classes...")

device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"   Dispositivo: {device}")

# ResNet18 pré-treinado
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Descongelar mais camadas para dataset pequeno (fine-tuning mais profundo)
for param in model.parameters():
    param.requires_grad = False

# Descongelar layer3, layer4 e fc
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Nova camada final para 10 classes
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Regularização
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

model = model.to(device)

# Loss com pesos de classe para desbalanceamento
class_weights_tensor = torch.tensor(
    [1.0 / class_counts_train.get(i, 1) for i in range(num_classes)],
    dtype=torch.float32
).to(device)
class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * num_classes

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['learning_rate'],
    weight_decay=0.01
)

# Scheduler com warmup
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG['learning_rate'] * 10,
    epochs=CONFIG['num_epochs'],
    steps_per_epoch=len(train_loader)
)

print(f"   Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %%
# ==================== 6. TREINAMENTO ====================
print(f"\n🚀 Iniciando treinamento ({CONFIG['num_epochs']} épocas)...")
print("=" * 65)

best_acc = 0.0
best_epoch = 0
history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_loss': []}
patience = 10
patience_counter = 0

for epoch in range(CONFIG['num_epochs']):
    # --- Treino ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # --- Validação ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    # Histórico
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Log
    lr = optimizer.param_groups[0]['lr']
    print(f"Época {epoch+1:02d}/{CONFIG['num_epochs']} | "
          f"Loss: {train_loss:.4f} | Train: {train_acc:.1f}% | "
          f"Val: {val_acc:.1f}% | LR: {lr:.6f}")
    
    # Salvar melhor
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        checkpoint = BASE_DIR / 'atlasleaf_v2_best_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names
        }, checkpoint)
        print(f"   ⭐ Novo melhor modelo! (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping na época {epoch+1}")
            break

print("\n" + "=" * 65)
print(f"✅ Treinamento concluído!")
print(f"   Melhor época: {best_epoch} | Melhor Val Acc: {best_acc:.2f}%")

# %%
# ==================== 7. EXPORT ONNX ====================
print("\n📦 Exportando modelo para ONNX...")

# Carregar melhor modelo
checkpoint = torch.load(BASE_DIR / 'atlasleaf_v2_best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model_cpu = model.to('cpu')

# Export
dummy_input = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size'])
onnx_path = BASE_DIR / CONFIG['output_onnx']

torch.onnx.export(
    model_cpu,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=[metadata['input_name']],
    output_names=[metadata['output_name']],
    dynamo=False
)

print(f"✅ Modelo ONNX exportado: {onnx_path}")

# %%
# ==================== 8. VALIDAÇÃO ONNX ====================
print("\n🔍 Validando modelo ONNX...")

onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print("   ✅ Estrutura ONNX válida")

ort_session = ort.InferenceSession(str(onnx_path))
test_input = np.random.randn(1, 3, CONFIG['img_size'], CONFIG['img_size']).astype(np.float32)
ort_outputs = ort_session.run(None, {metadata['input_name']: test_input})

print(f"   ✅ Inferência ONNX funcional")
print(f"   Output shape: {ort_outputs[0].shape} ({num_classes} classes)")

# %%
# ==================== 9. TESTE E MÉTRICAS ====================
print("\n📊 Avaliação detalhada no conjunto de validação...")

model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Acurácia por classe
print("\n   Acurácia por Doença:")
for i, cls_name in enumerate(class_names):
    cls_mask = np.array(all_labels) == i
    if cls_mask.sum() > 0:
        cls_acc = (np.array(all_preds)[cls_mask] == i).mean() * 100
        friendly = metadata['friendly_names'].get(cls_name, cls_name)
        print(f"   - {friendly}: {cls_acc:.1f}%")

# %%
# ==================== 10. RESUMO FINAL ====================
print("\n" + "=" * 70)
print("🎉 ATLASLEAF v2.0 - TREINAMENTO CONCLUÍDO!")
print("=" * 70)

print(f"\n📊 Resumo:")
print(f"   - Versão: 2.0 (Multi-Classe)")
print(f"   - Classes: {num_classes} doenças específicas")
print(f"   - Melhor Acurácia: {best_acc:.2f}%")
print(f"   - Épocas treinadas: {best_epoch}")

print(f"\n🦠 Doenças Detectáveis:")
for cls in class_names:
    friendly = metadata['friendly_names'].get(cls, cls)
    print(f"   • {friendly}")

print(f"\n📁 Arquivos Gerados:")
print(f"   1. {CONFIG['output_onnx']}")
print(f"   2. atlasleaf_v2_metadata.json")
print(f"   3. atlasleaf_v2_best_model.pth")

print("\n" + "=" * 70)
