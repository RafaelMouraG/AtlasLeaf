"""
AtlasLeaf v3.1 - Configurações de Treinamento
============================================
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class OversamplingConfig:
    """Configuração para oversampling adaptativo de classes minoritárias."""
    
    # Thresholds para determinar o factor de oversampling
    rare_threshold: int = 100      # Classe rara: < 100 amostras
    medium_threshold: int = 300    # Classe média: 100-300 amostras
    
    # Fatores de oversampling
    rare_factor: int = 4           # 4x para classes raras
    medium_factor: int = 2         # 2x para classes médias
    common_factor: int = 1         # 1x para classes comuns
    
    # Intensidade de augmentação (para uso futuro)
    rare_aug_intensity: str = "strong"
    medium_aug_intensity: str = "medium"
    common_aug_intensity: str = "light"
    
    def get_config_for_class(self, count: int) -> Tuple[int, str]:
        """
        Retorna (factor, aug_intensity) baseado na contagem da classe.
        
        Args:
            count: Número de amostras da classe
            
        Returns:
            Tuple de (factor de oversampling, intensidade de augmentação)
        """
        if count < self.rare_threshold:
            return self.rare_factor, self.rare_aug_intensity
        elif count < self.medium_threshold:
            return self.medium_factor, self.medium_aug_intensity
        else:
            return self.common_factor, self.common_aug_intensity


@dataclass
class TrainingConfigV31:
    """Configuração completa para treinamento AtlasLeaf v3.1."""
    
    # Modelo
    model_name: str = "efficientnet_v2_s"  # 🔄 Trocado de B3 para V2-S
    num_classes: int = 15
    dropout: float = 0.5
    pretrained: bool = True
    
    # Input
    input_size: int = 384  # V2-S funciona bem com 384
    
    # Treinamento
    epochs: int = 100
    batch_size: int = 12
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    
    # Gradient Accumulation
    accumulation_steps: int = 4
    
    @property
    def effective_batch_size(self) -> int:
        """Batch size efetivo considerando gradient accumulation."""
        return self.batch_size * self.accumulation_steps
    
    # Scheduler - OneCycleLR
    onecycle_pct_start: float = 0.1  # 10% warmup
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 10000.0
    
    # Early Stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Focal Loss
    focal_gamma_start: float = 1.0
    focal_gamma_end: float = 3.0
    focal_gamma_warmup_epochs: int = 20
    
    # Label Smoothing
    label_smoothing: float = 0.1
    
    # Mixup / CutMix
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 0.4
    
    # Augmentation
    augmentation_strength: str = "strong"  # light, medium, strong
    
    def get_focal_gamma(self, epoch: int, total_epochs: int) -> float:
        """
        Calcula gamma do Focal Loss com warmup progressivo.
        
        Args:
            epoch: Época atual
            total_epochs: Total de épocas
            
        Returns:
            Valor de gamma para a época atual
        """
        if epoch < self.focal_gamma_warmup_epochs:
            # Warmup linear
            progress = epoch / self.focal_gamma_warmup_epochs
            return self.focal_gamma_start + (self.focal_gamma_end - self.focal_gamma_start) * progress
        return self.focal_gamma_end
    
    def get_device(self) -> torch.device:
        """Retorna o melhor device disponível."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


@dataclass 
class M3ProConfig(TrainingConfigV31):
    """Configuração otimizada para MacBook Pro M3 Pro (18GB RAM unificada)."""
    
    # Otimizações para Apple Silicon
    batch_size: int = 16           # Maior batch para M3 Pro
    accumulation_steps: int = 2    # Menos accumulation necessário
    
    # Ajustes para treino mais rápido no M3
    epochs: int = 80
    learning_rate: float = 8e-4    # LR ligeiramente maior
    
    # Early stopping mais agressivo
    early_stopping_patience: int = 12


def print_system_info():
    """Printa informações do sistema."""
    import platform
    import torch
    
    print(f"💻 Sistema: {platform.system()} {platform.machine()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("🚀 Apple Silicon (MPS) detectado!")
    elif torch.cuda.is_available():
        print(f"🚀 CUDA disponível: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Usando CPU")
