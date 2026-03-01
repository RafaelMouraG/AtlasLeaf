"""
AtlasLeaf v3.1 - Script de Treinamento Otimizado
================================================

Melhorias:
1. EfficientNet-B3 (melhor que ResNet50)
2. Focal Loss adaptativo
3. Mixup + CutMix
4. Gradient Accumulation
5. OneCycleLR
6. Validação cruzada estratificada
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Importa módulos locais
sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline.model_v31 import create_model, CombinedLoss, mixup_data, cutmix_data, mixup_criterion
from data_pipeline.config_v31 import TrainingConfigV31, OversamplingConfig


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

def set_seed(seed: int = 42):
    """Define seed para reproducibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Retorna device disponível."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DATASET COM OVERSAMPLING
# =============================================================================

class SoybeanDataset(Dataset):
    """
    Dataset com oversampling adaptativo para classes minoritárias.
    """
    
    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform=None,
        oversampling_config: OversamplingConfig = None,
    ):
        self.transform = transform
        self.oversampling_config = oversampling_config or OversamplingConfig()
        
        # Conta amostras por classe
        class_counts = Counter(labels)
        
        # Aplica oversampling
        self.samples = []
        for path, label in zip(image_paths, labels):
            count = class_counts[label]
            factor, aug_intensity = self.oversampling_config.get_config_for_class(count)
            
            # Adiciona múltiplas cópias para classes minoritárias
            for _ in range(factor):
                self.samples.append((path, label, aug_intensity))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, aug_intensity = self.samples[idx]
        
        # Carrega imagem
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Erro carregando {path}: {e}")
            image = Image.new('RGB', (384, 384), color='black')
        
        # Aplica transformação
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(
    size: int = 384,
    is_training: bool = True,
    augmentation_strength: str = "medium",
):
    """Retorna transformações para treino ou validação."""
    
    if not is_training:
        # Validação/Teste - apenas normalização
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Treino - augmentation baseado na força
    if augmentation_strength == "light":
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    elif augmentation_strength == "medium":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    else:  # strong
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])


# =============================================================================
# FUNÇÕES DE TREINAMENTO
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: TrainingConfigV31,
    epoch: int,
    use_mixup: bool = True,
    use_cutmix: bool = True,
) -> tuple:
    """Treina por uma época com gradient accumulation."""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixup ou CutMix
        mixed_images = images
        labels_a = labels
        labels_b = None
        lam = 1.0
        use_aug = False
        
        if use_mixup and random.random() < 0.5:
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, config.mixup_alpha)
            use_aug = True
        elif use_cutmix and random.random() < 0.5:
            mixed_images, labels_a, labels_b, lam = cutmix_data(images, labels, config.cutmix_alpha)
            use_aug = True
        
        # Forward
        outputs = model(mixed_images)
        
        # Loss
        if use_aug and labels_b is not None:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        # Normaliza loss pelo número de accumulation steps
        loss = loss / config.accumulation_steps
        loss.backward()
        
        # Accumulation
        if (batch_idx + 1) % config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Métricas
        total_loss += loss.item() * config.accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        if use_aug and labels_b is not None:
            correct += (lam * predicted.eq(labels_a).sum().item() + 
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Valida o modelo."""
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(loader),
        100. * correct / total,
        all_preds,
        all_labels
    )


# =============================================================================
# TREINAMENTO COM VALIDAÇÃO CRUZADA
# =============================================================================

def train_with_cross_validation(
    image_paths: list,
    labels: list,
    config: TrainingConfigV31,
    device: torch.device,
    n_folds: int = 5,
):
    """Treina com validação cruzada estratificada."""
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n{'='*60}")
        print(f"🔄 Fold {fold + 1}/{n_folds}")
        print('='*60)
        
        # Split dados
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Contagem por classe
        class_counts = [0] * config.num_classes
        for l in train_labels:
            class_counts[l] += 1
        
        print(f"📊 Classes no treino: {class_counts}")
        
        # Datasets
        train_dataset = SoybeanDataset(
            train_paths,
            train_labels,
            transform=get_transforms(config.input_size, is_training=True, augmentation_strength="strong"),
            oversampling_config=OversamplingConfig(),
        )
        
        val_dataset = SoybeanDataset(
            val_paths,
            val_labels,
            transform=get_transforms(config.input_size, is_training=False),
        )
        
        # Samplers
        train_sampler = WeightedRandomSampler(
            weights=[1.0 / class_counts[l] for _, l in train_dataset.samples],
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=0,  # MPS não suporta multiprocessing
            pin_memory=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        # Modelo
        model = create_model(
            model_name=config.model_name,
            num_classes=config.num_classes,
            pretrained=True,
            dropout=config.dropout,
        ).to(device)
        
        # Loss com Focal Loss adaptativo
        criterion = CombinedLoss(
            num_classes=config.num_classes,
            class_counts=class_counts,
            gamma=config.focal_gamma_start,
            smoothing=config.label_smoothing,
        )
        
        # Otimizador
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler - OneCycleLR
        steps_per_epoch = len(train_loader) // config.accumulation_steps
        total_steps = steps_per_epoch * config.epochs
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )
        
        # Treinamento
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(config.epochs):
            # Atualiza gamma do Focal Loss
            current_gamma = config.get_focal_gamma(epoch, config.epochs)
            criterion.gamma = current_gamma
            
            # Treino
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, config, epoch
            )
            
            # Validação
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}% | Gamma={current_gamma:.2f}")
            
            # Early stopping
            if val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Salva melhor modelo do fold
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, f'atlasleaf_v31_fold{fold+1}_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'best_val_acc': max(history['val_acc']),
            'history': history,
        })
    
    return fold_results


# =============================================================================
# MAIN
# =============================================================================

def load_dataset(dataset_dir: Path):
    """Carrega dataset unificado."""
    manifest_path = dataset_dir / "manifest.json"
    splits_path = dataset_dir / "splits.json"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)
    
    # Carrega todos os dados para CV
    all_images = []
    all_labels = []
    
    path_to_class = {img["path"]: img["class_id"] for img in manifest["images"]}
    
    for split in ["train", "val", "test"]:
        for rel_path in splits[split]:
            if rel_path in path_to_class:
                full_path = dataset_dir / rel_path
                all_images.append(str(full_path))
                all_labels.append(path_to_class[rel_path])
    
    return all_images, all_labels


def main():
    parser = argparse.ArgumentParser(description='Treina AtlasLeaf v3.1')
    parser.add_argument('--data-dir', type=str, default='datasets/unified',
                       help='Diretório do dataset')
    parser.add_argument('--model', type=str, default='efficientnet_b3',
                       choices=['efficientnet_b3', 'convnext_tiny', 'mobilenet_v3', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--folds', type=int, default=5,
                       help='Número de folds para validação cruzada (0 = sem CV)')
    parser.add_argument('--no-cv', action='store_true',
                       help='Desabilitar validação cruzada')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    device = get_device()
    print(f"🖥️  Device: {device}")
    print(f"🔢 PyTorch: {torch.__version__}")
    
    # Config
    config = TrainingConfigV31(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    print(f"\n⚙️  Configuração:")
    print(f"   Modelo: {config.model_name}")
    print(f"   Batch: {config.batch_size} (efetivo: {config.effective_batch_size})")
    print(f"   LR: {config.learning_rate}")
    print(f"   Epochs: {config.epochs}")
    
    # Carrega dados
    dataset_dir = Path(args.data_dir)
    image_paths, labels = load_dataset(dataset_dir)
    
    print(f"\n📊 Dataset:")
    print(f"   Total: {len(image_paths)} imagens")
    print(f"   Classes: {len(set(labels))}")
    
    # Contagem por classe
    class_dist = Counter(labels)
    print("\n   Distribuição:")
    for cls_id, count in sorted(class_dist.items()):
        print(f"     Classe {cls_id}: {count} amostras")
    
    # Treinamento
    if not args.no_cv and args.folds > 1:
        print(f"\n🔄 Iniciando validação cruzada ({args.folds} folds)...")
        results = train_with_cross_validation(
            image_paths, labels, config, device, n_folds=args.folds
        )
        
        # Resultados agregados
        print(f"\n{'='*60}")
        print("📊 Resultados da Validação Cruzada:")
        print('='*60)
        
        val_accs = [r['best_val_acc'] for r in results]
        val_losses = [r['best_val_loss'] for r in results]
        
        print(f"Acurácia média: {np.mean(val_accs):.2f}% (+/- {np.std(val_accs):.2f})")
        print(f"Loss média: {np.mean(val_losses):.4f} (+/- {np.std(val_losses):.4f})")
        
        for r in results:
            print(f"  Fold {r['fold']}: Acc={r['best_val_acc']:.2f}%, Loss={r['best_val_loss']:.4f}")
    
    else:
        print("\n🚀 Treinamento simples (sem CV)...")
        # Implementação simplificada sem CV
        # (código similar ao train_with_cross_validation mas sem o loop de folds)
        pass


if __name__ == "__main__":
    main()
