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
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
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


class SimpleSoybeanDataset(Dataset):
    """Dataset simples, sem oversampling, para validação/teste."""

    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path, label = self.image_paths[idx], self.labels[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Erro carregando {path}: {e}")
            image = Image.new('RGB', (384, 384), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(
    size: int = 384,
    is_training: bool = True,
    augmentation_strength: str = "medium",
    leaf_crop: bool = False,
    leaf_crop_mask_bg: bool = False,
    domain_aug: bool = False,
):
    """Retorna transformações para treino ou validação.

    Se leaf_crop=True, aplica LeafCropper (isola a folha) ANTES do resize.
    Use isto quando treinar com imagens NÃO pré-recortadas. Se você já rodou
    preprocess_leaf_crop.py e aponta para datasets/unified_cropped, deixe False.

    Se domain_aug=True (só treino), adiciona randomização de domínio (resolução +
    JPEG) e reforça o ColorJitter para apagar a assinatura de fonte nos pixels.
    """

    pre = []
    if leaf_crop:
        from data_pipeline.leaf_segmentation import LeafCropper
        pre = [LeafCropper(mask_background=leaf_crop_mask_bg)]

    if domain_aug and is_training:
        from data_pipeline.domain_aug import domain_transforms
        pre = pre + domain_transforms()

    if not is_training:
        # Validação/Teste - apenas normalização
        return transforms.Compose(pre + [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Treino - augmentation baseado na força
    if augmentation_strength == "light":
        return transforms.Compose(pre + [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    elif augmentation_strength == "medium":
        return transforms.Compose(pre + [
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
        # domain_aug reforça a cor (ataca a ciência de cor da câmera)
        jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.15) if domain_aug \
                 else transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
        return transforms.Compose(pre + [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            jitter,
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
    scheduler,
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
        
        # CORREÇÃO: Acumula loss ANTES de normalizar para relatório
        loss_value = loss.item()
        total_loss += loss_value
        
        # Normaliza loss pelo número de accumulation steps para backward
        loss = loss / config.accumulation_steps
        loss.backward()
        
        # Accumulation
        if (batch_idx + 1) % config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # CORREÇÃO: OneCycleLR precisa ser chamado a cada batch
            optimizer.zero_grad()
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
        
        val_dataset = SimpleSoybeanDataset(
            val_paths,
            val_labels,
            transform=get_transforms(config.input_size, is_training=False),
        )
        
        # Samplers
        # CORREÇÃO: Desempacotamento correto da tupla (path, label, aug_intensity)
        train_sampler = WeightedRandomSampler(
            weights=[1.0 / class_counts[label] for _, label, _ in train_dataset.samples],
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
                model, train_loader, criterion, optimizer, scheduler, device, config, epoch
            )
            
            # Validação
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
            
            # NOTA: scheduler.step() foi movido para dentro do batch (OneCycleLR)
            
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
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"❌ Manifest não encontrado: {manifest_path}\n"
            f"   Execute primeiro: python data_pipeline/dataset_unifier.py"
        )
    
    if not splits_path.exists():
        raise FileNotFoundError(
            f"❌ Splits não encontrado: {splits_path}\n"
            f"   Execute primeiro: python data_pipeline/dataset_unifier.py"
        )
    
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


def load_presplit_dataset(dataset_dir: Path, split_file: str = "splits_source.json"):
    """
    Carrega um dataset já dividido (ex.: split por fonte, gerado por
    data_pipeline/source_split.py). Retorna dict split -> (paths, labels).

    Diferente de load_dataset(), NÃO junta tudo: preserva train/val/test para
    avaliação honesta (sem re-embaralhar e revasar a fonte).
    """
    manifest = json.load(open(dataset_dir / "manifest.json"))
    splits = json.load(open(dataset_dir / split_file))
    path_to_class = {img["path"]: int(str(img["class_id"])) for img in manifest["images"]}

    out = {}
    for split in ["train", "val", "test", "test_insource"]:
        paths, labels = [], []
        for rel_path in splits.get(split, []):
            if rel_path in path_to_class:
                # tolera troca de extensão feita pelo preprocess (.jpg)
                fp = dataset_dir / rel_path
                if not fp.exists():
                    alt = fp.with_suffix(".jpg")
                    if alt.exists():
                        fp = alt
                paths.append(str(fp))
                labels.append(path_to_class[rel_path])
        out[split] = (paths, labels)
    return out


def train_holdout(
    data: dict,
    config: TrainingConfigV31,
    device: torch.device,
    leaf_crop: bool = False,
    leaf_crop_mask_bg: bool = False,
    domain_aug: bool = False,
    freeze_backbone: bool = False,
    class_names: dict = None,
    ckpt_out: str = "atlasleaf_v31_sourcesplit_best.pth",
):
    """
    Treina respeitando o split por fonte: treina em `train`, valida em `val`,
    e avalia `test` (cross-source, HONESTO) e `test_insource` (otimista) à parte.
    """
    train_paths, train_labels = data["train"]
    val_paths, val_labels = data["val"]

    class_counts = [0] * config.num_classes
    for l in train_labels:
        class_counts[l] += 1
    print(f"📊 Amostras de treino por classe: {class_counts}")

    # IMPORTANTE: UMA rebalanceação suave (sqrt-inverso via sampler), sem empilhar.
    # Antes havia compensação TRIPLA (oversampling x sampler 1/count x focal weights),
    # que zerava as classes grandes -> 0% recall nas majoritárias. Aqui:
    #   - sem duplicação (SimpleSoybeanDataset no treino)
    #   - sampler com peso 1/sqrt(count) (rebalanceia sem colapsar as majoritárias)
    #   - focal loss SEM peso de classe (class_counts=None) -> só o foco em exemplos difíceis
    train_dataset = SimpleSoybeanDataset(
        train_paths, train_labels,
        transform=get_transforms(config.input_size, is_training=True,
                                 augmentation_strength="strong",
                                 leaf_crop=leaf_crop, leaf_crop_mask_bg=leaf_crop_mask_bg,
                                 domain_aug=domain_aug),
    )
    eval_tf = get_transforms(config.input_size, is_training=False,
                             leaf_crop=leaf_crop, leaf_crop_mask_bg=leaf_crop_mask_bg)
    val_dataset = SimpleSoybeanDataset(val_paths, val_labels, transform=eval_tf)

    sample_weights = [1.0 / (max(1, class_counts[l]) ** 0.5) for l in train_labels]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler=train_sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = create_model(config.model_name, config.num_classes, pretrained=True,
                         dropout=config.dropout).to(device)

    # Opção: congelar o backbone e treinar só a cabeça (classifier). Preserva as
    # features transferíveis do ImageNet -> costuma generalizar MELHOR cross-source
    # do que fine-tune completo (que distorce as features p/ as fontes de treino).
    if freeze_backbone:
        n_frozen = 0
        for p in model.parameters():
            p.requires_grad = False
            n_frozen += 1
        head = model.classifier if hasattr(model, "classifier") else model.fc
        for p in head.parameters():
            p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"🧊 Backbone congelado: treinando só a cabeça ({sum(p.numel() for p in trainable):,} params)")
        params = trainable
    else:
        params = model.parameters()

    criterion = CombinedLoss(config.num_classes, class_counts=None,
                             gamma=config.focal_gamma_start, smoothing=config.label_smoothing)
    optimizer = optim.AdamW(params, lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // config.accumulation_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=steps_per_epoch * config.epochs,
        pct_start=0.1, anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0)

    # Early stopping por ACURÁCIA BALANCEADA da validação (val_loss ponderada engana).
    best_val_bal = -1.0
    patience = 0
    for epoch in range(config.epochs):
        criterion.gamma = config.get_focal_gamma(epoch, config.epochs)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer,
                                            scheduler, device, config, epoch)
        val_loss, val_acc, val_preds, val_labs = validate(model, val_loader, criterion, device)
        val_bal = 100.0 * balanced_accuracy_score(val_labs, val_preds)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}% BalAcc={val_bal:.2f}%")
        if val_bal > best_val_bal + config.early_stopping_min_delta:
            best_val_bal = val_bal
            patience = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc, 'val_bal_acc': val_bal},
                       ckpt_out)
        else:
            patience += 1
        if patience >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} (melhor BalAcc={best_val_bal:.2f}%)")
            break

    # Recarrega o MELHOR checkpoint (não o último estado pós early-stopping)
    try:
        ckpt = torch.load(ckpt_out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\n🔁 Avaliando melhor checkpoint (epoch {ckpt.get('epoch','?')+1 if isinstance(ckpt.get('epoch'),int) else '?'}, "
              f"BalAcc val={ckpt.get('val_bal_acc',0):.2f}%)")
    except Exception as e:
        print(f"\n⚠️ Não recarregou checkpoint ({e}); avaliando modelo em memória.")

    # Avaliação final honesta
    for split_name, tag in [("test", "HELD-OUT / HONESTO (fonte ou câmera não vista)"),
                            ("test_insource", "IN-SOURCE (otimista)")]:
        paths, labels = data.get(split_name, ([], []))
        if not paths:
            continue
        ds = SimpleSoybeanDataset(paths, labels, transform=eval_tf)
        loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        _, acc, preds, labs = validate(model, loader, criterion, device)
        bal = 100.0 * balanced_accuracy_score(labs, preds)
        print(f"\n=== Teste {tag}: acc={acc:.2f}%  BalAcc={bal:.2f}%  (n={len(labels)}) ===")
        present = sorted(set(labs))
        names = [class_names.get(c, str(c)) if class_names else str(c) for c in present]
        print(classification_report(labs, preds, labels=present,
                                    target_names=names, zero_division=0))
    return model


def main():
    parser = argparse.ArgumentParser(description='Treina AtlasLeaf v3.1')
    parser.add_argument('--data-dir', type=str, default='datasets/unified',
                       help='Diretório do dataset')
    parser.add_argument('--model', type=str, default='efficientnet_v2_s',
                       choices=['efficientnet_v2_s', 'efficientnet_b3', 'convnext_tiny', 'mobilenet_v3', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num-classes', type=int, default=15,
                       help='Nº de saídas do modelo (aumente se o dataset combinado tiver doenças novas)')
    parser.add_argument('--folds', type=int, default=5,
                       help='Número de folds para validação cruzada (0 = sem CV)')
    parser.add_argument('--no-cv', action='store_true',
                       help='Desabilitar validação cruzada')
    parser.add_argument('--source-split', action='store_true',
                       help='Treina respeitando splits_source.json (avaliação honesta cross-source)')
    parser.add_argument('--split-file', type=str, default='splits_source.json')
    parser.add_argument('--leaf-crop', action='store_true',
                       help='Aplica LeafCropper on-the-fly (use se as imagens NÃO forem pré-recortadas)')
    parser.add_argument('--leaf-crop-mask-bg', action='store_true',
                       help='LeafCropper pinta o fundo de cinza (mais agressivo)')
    parser.add_argument('--domain-aug', action='store_true',
                       help='Augmentation de domínio (resolução+JPEG+cor) p/ apagar assinatura de fonte')
    parser.add_argument('--ckpt-out', type=str, default='atlasleaf_v31_sourcesplit_best.pth',
                       help='Arquivo do melhor checkpoint (use um nome diferente em testes!)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Congela o backbone e treina só a cabeça (melhor generalização cross-source)')

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
    config.num_classes = args.num_classes
    
    print(f"\n⚙️  Configuração:")
    print(f"   Modelo: {config.model_name}")
    print(f"   Batch: {config.batch_size} (efetivo: {config.effective_batch_size})")
    print(f"   LR: {config.learning_rate}")
    print(f"   Epochs: {config.epochs}")
    
    dataset_dir = Path(args.data_dir)

    # Caminho HONESTO: split por fonte + (opcional) recorte de folha
    if args.source_split:
        print(f"\n🧪 Modo split-por-fonte ({args.split_file}) — avaliação honesta")
        manifest = json.load(open(dataset_dir / "manifest.json"))
        class_names = {v["id"]: k for k, v in manifest["class_distribution"].items()}
        data = load_presplit_dataset(dataset_dir, args.split_file)
        for k, (p, _) in data.items():
            print(f"   {k}: {len(p)} imagens")
        train_holdout(data, config, device,
                      leaf_crop=args.leaf_crop, leaf_crop_mask_bg=args.leaf_crop_mask_bg,
                      domain_aug=args.domain_aug, freeze_backbone=args.freeze_backbone,
                      class_names=class_names, ckpt_out=args.ckpt_out)
        return

    # Carrega dados
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
