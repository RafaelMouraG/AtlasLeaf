"""
AtlasLeaf v3.1 - Modelos e Loss Functions
=========================================

Inclui:
- EfficientNet-B3, ConvNeXt Tiny, MobileNet V3, ResNet50
- Focal Loss e CombinedLoss
- Mixup e CutMix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Tuple
import random


def create_model(
    model_name: str = "efficientnet_b3",
    num_classes: int = 15,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Cria modelo de classificação com backbone selecionado.
    
    Args:
        model_name: Nome do modelo ('efficientnet_v2_s', 'efficientnet_b3', 'convnext_tiny', 'mobilenet_v3', 'resnet50')
        num_classes: Número de classes de saída
        pretrained: Se True, usa pesos ImageNet
        dropout: Taxa de dropout para regularização
        
    Returns:
        Modelo PyTorch pronto para treinamento
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    weights_v2 = "IMAGENET1K_V1" if pretrained else None
    
    if model_name == "efficientnet_v2_s":
        # 🆕 EfficientNet-V2-S - mais moderno e eficiente
        model = models.efficientnet_v2_s(weights=weights_v2)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        # Modifica classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes)
        )
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")
    
    return model


class FocalLoss(nn.Module):
    """
    Focal Loss para lidar com desequilíbrio de classes.
    
    Foca em exemplos difíceis e reduz o peso de exemplos fáceis.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_counts: Optional[List[int]] = None,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
        # Calcula pesos inversos baseados nas contagens
        if class_counts is not None:
            total = sum(class_counts)
            # Peso inverso: classes raras têm peso maior
            weights = [total / (len(class_counts) * count) if count > 0 else 1.0 
                      for count in class_counts]
            # Normaliza para média 1
            mean_weight = sum(weights) / len(weights)
            weights = [w / mean_weight for w in weights]
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.weights = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Garante que os pesos estejam no mesmo device que os inputs
        weights = self.weights.to(inputs.device) if self.weights is not None else None

        if self.label_smoothing > 0 and inputs.device.type == 'mps':
            n_classes = inputs.size(1)
            log_probs = F.log_softmax(inputs, dim=1)
            with torch.no_grad():
                targets_one_hot = torch.zeros_like(log_probs)
                targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
                targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                    self.label_smoothing / n_classes
            ce_loss = -(targets_smooth * log_probs).sum(dim=1)
            if weights is not None:
                ce_loss = ce_loss * weights[targets]  # já está no device certo
        else:
            ce_loss = F.cross_entropy(
                inputs, targets,
                weight=weights,  # agora no device correto
                label_smoothing=self.label_smoothing,
                reduction='none'
            )

        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        return (focal_weight * ce_loss).mean()


class CombinedLoss(nn.Module):
    """
    Combina Focal Loss com regularização.
    
    Permite ajuste dinâmico do gamma durante o treinamento.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_counts: Optional[List[int]] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        
        self.focal = FocalLoss(
            num_classes=num_classes,
            class_counts=class_counts,
            gamma=gamma,
            label_smoothing=smoothing,
        )
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Atualiza gamma dinamicamente
        self.focal.gamma = self.gamma
        return self.focal(inputs, targets)


# =============================================================================
# MIXUP E CUTMIX
# =============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Aplica Mixup nas imagens.
    
    Mixup: lambda * x_i + (1 - lambda) * x_j
    
    Args:
        x: Batch de imagens [N, C, H, W]
        y: Labels [N]
        alpha: Parâmetro da distribuição Beta
        
    Returns:
        (mixed_x, y_a, y_b, lambda)
    """
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Aplica CutMix nas imagens.
    
    CutMix: substitui uma região aleatória de x_i com x_j.
    
    Args:
        x: Batch de imagens [N, C, H, W]
        y: Labels [N]
        alpha: Parâmetro da distribuição Beta
        
    Returns:
        (mixed_x, y_a, y_b, lambda)
    """
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    _, _, h, w = x.shape
    
    # Calcula tamanho do box
    cut_ratio = (1 - lam) ** 0.5
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    # Posição aleatória
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)
    
    # Aplica cutmix
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Ajusta lambda proporcional à área
    lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Calcula loss para Mixup/CutMix.
    
    loss = lambda * loss(y_a) + (1 - lambda) * loss(y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# TESTE
# =============================================================================

if __name__ == "__main__":
    print("Testando modelos...")
    
    # Testa criação de modelos
    for model_name in ["efficientnet_v2_s", "efficientnet_b3", "convnext_tiny", "mobilenet_v3", "resnet50"]:
        model = create_model(model_name, num_classes=15)
        print(f"✅ {model_name}: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Testa Mixup
    x = torch.randn(8, 3, 384, 384)
    y = torch.randint(0, 15, (8,))
    
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    print(f"\n✅ Mixup: lam={lam:.3f}")
    
    # Testa CutMix
    mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=0.4)
    print(f"✅ CutMix: lam={lam:.3f}")
    
    # Testa FocalLoss
    criterion = FocalLoss(num_classes=15, gamma=2.0)
    logits = torch.randn(8, 15)
    loss = criterion(logits, y)
    print(f"✅ FocalLoss: {loss.item():.4f}")
