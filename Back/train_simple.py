#!/usr/bin/env python3
"""
🌿 AtlasLeaf v3.1 - Treinamento Simples (Modo "Play Button")
===========================================================

Versão simplificada para quem quer só clicar em "Run" e ver funcionar.
Sem argumentos, sem complicação.

Uso:
    python train_simple.py

Isso vai:
    1. Detectar seu M3 Pro e usar config otimizada
    2. Carregar o dataset automaticamente
    3. Treinar com as melhores práticas
    4. Salvar o modelo pronto para uso
"""

import sys
import os
from pathlib import Path

# Configura paths
BACK_DIR = Path(__file__).parent
sys.path.insert(0, str(BACK_DIR))

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
from datetime import datetime
from collections import Counter
from tqdm import tqdm
import numpy as np

# Configurações do AtlasLeaf
from data_pipeline.config_m3pro import M3ProConfig, print_system_info
from data_pipeline.model_v31 import create_model


def print_header(text):
    """Print formatado."""
    print("\n" + "="*60)
    print(f"🌿 {text}")
    print("="*60)


def load_dataset_simple(dataset_dir):
    """Carrega dataset de forma simples."""
    manifest_path = dataset_dir / "manifest.json"
    splits_path = dataset_dir / "splits.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"❌ Dataset não encontrado em: {dataset_dir}\n"
            f"   Execute primeiro: python data_pipeline/dataset_unifier.py"
        )
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)
    
    # Carrega apenas treino e val (deixa teste para depois)
    path_to_class = {img["path"]: img["class_id"] for img in manifest["images"]}
    
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    
    for rel_path in splits["train"]:
        if rel_path in path_to_class:
            train_images.append(str(dataset_dir / rel_path))
            train_labels.append(path_to_class[rel_path])
    
    for rel_path in splits["val"]:
        if rel_path in path_to_class:
            val_images.append(str(dataset_dir / rel_path))
            val_labels.append(path_to_class[rel_path])
    
    return (train_images, train_labels), (val_images, val_labels)


class SimpleDataset(torch.utils.data.Dataset):
    """Dataset simplificado."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(path).convert('RGB')
        except:
            image = Image.new('RGB', (384, 384), 'black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_epoch_simple(model, loader, criterion, optimizer, device):
    """Treina uma época (versão simples)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def validate_simple(model, loader, criterion, device):
    """Valida (versão simples)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    """Função principal - só rodar e acompanhar."""
    
    # 1. Info do sistema
    print_header("AtlasLeaf v3.1 - Treinamento Simplificado")
    print_system_info()
    
    # 2. Configuração
    config = M3ProConfig()
    print(f"\n⚙️  Configuração: {config.model_name}")
    print(f"   Batch: {config.batch_size} | Épocas: {config.epochs}")
    print(f"   Resolução: {config.input_size}x{config.input_size}")
    
    # 3. Device
    device = config.get_device()
    print(f"\n🖥️  Usando device: {device}")
    
    # 4. Carrega dataset
    print_header("Carregando Dataset")
    dataset_dir = BACK_DIR / "datasets" / "unified"
    
    try:
        (train_paths, train_labels), (val_paths, val_labels) = load_dataset_simple(dataset_dir)
        print(f"✅ Train: {len(train_paths)} imagens")
        print(f"✅ Val: {len(val_paths)} imagens")
        print(f"✅ Classes: {len(set(train_labels))}")
    except FileNotFoundError as e:
        print(e)
        print("\n💡 Para criar o dataset, execute:")
        print("   python data_pipeline/dataset_unifier.py")
        return
    
    # 5. DataLoaders com Augmentação FORTE (para generalização em fotos de campo)
    print_header("Preparando DataLoaders")
    print("⚡ Usando augmentações agressivas para melhor generalização em campo")
    
    # Importa DomainRandomization para simular condições de campo
    from data_pipeline.augmentation import DomainRandomization
    
    # Augmentação PIL (antes de ToTensor)
    class FieldAugmentation:
        """Augmentação para simular fotos de campo."""
        def __init__(self):
            self.domain_random = DomainRandomization(p=0.3)
        
        def __call__(self, img):
            return self.domain_random(img)
    
    train_transform = transforms.Compose([
        # Crop aleatório para diferentes escalas/enquadramentos
        transforms.RandomResizedCrop(
            config.input_size, 
            scale=(0.7, 1.0),  # Permite zoom in/out
            ratio=(0.8, 1.2)   # Permite deformação
        ),
        # Flips
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        # Rotação mais agressiva
        transforms.RandomRotation(45),
        # Domain Randomization (sombras, luz solar, etc)
        FieldAugmentation(),
        # Variação de cor mais forte
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3, 
            saturation=0.3,
            hue=0.1  # Variação de matiz
        ),
        # Blur ocasional (simula foco)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.2),
        # Grayscale ocasional
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        # Random Erasing (simula oclusões)
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = SimpleDataset(train_paths, train_labels, train_transform)
    val_dataset = SimpleDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    
    # 6. Modelo
    print_header("Criando Modelo")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=True,
        dropout=config.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo: {config.model_name}")
    print(f"   Parâmetros: {total_params:,}")
    
    # 7. Loss e Optimizer (simplificado - CrossEntropy com pesos)
    class_counts = [0] * config.num_classes
    for l in train_labels:
        class_counts[l] += 1
    
    # Calcula pesos inversos (simples e estável)
    total_samples = sum(class_counts)
    weights = [total_samples / (len(class_counts) * count) if count > 0 else 0.0 
               for count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"   Class weights: {[f'{w:.2f}' for w in weights]}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 8. Treinamento
    print_header("Iniciando Treinamento")
    print("   Pressione Ctrl+C para parar (salva checkpoint)\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(config.epochs):
            print(f"\n📚 Época {epoch+1}/{config.epochs}")
            print("-" * 60)
            
            # Treino
            train_loss, train_acc = train_epoch_simple(model, train_loader, criterion, optimizer, device)
            
            # Validação
            val_loss, val_acc = validate_simple(model, val_loader, criterion, device)
            
            # Scheduler
            scheduler.step(val_loss)
            
            # Log
            print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            # Salva melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config.__dict__,
                }, BACK_DIR / "atlasleaf_v31_best_model.pth")
                
                print(f"💾 Novo melhor modelo salvo! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config.early_stopping_patience:
                print(f"\n⛔ Early stopping após {epoch+1} épocas")
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Treino interrompido pelo usuário")
        print("   Salvando checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, BACK_DIR / "atlasleaf_v31_interrupted.pth")
    
    # 9. Finalização
    print_header("Treinamento Finalizado")
    print(f"✅ Melhor modelo: atlasleaf_v31_best_model.pth")
    print(f"✅ Para usar: streamlit run app_streamlit_v31.py")
    
    # Exporta para ONNX
    print("\n🔄 Exportando para ONNX...")
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, config.input_size, config.input_size).to(device)
        
        torch.onnx.export(
            model,
            dummy_input,
            BACK_DIR / "atlasleaf_v31_diseases.onnx",
            export_params=True,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,  # Usa exportador legado (AdaptiveMaxPool2d não suportado no dynamo)
        )
        print("✅ Modelo exportado: atlasleaf_v31_diseases.onnx")
    except Exception as e:
        print(f"⚠️  Erro ao exportar ONNX: {e}")


if __name__ == "__main__":
    main()
