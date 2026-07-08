"""
AtlasLeaf v3.1 - Pipeline de Inferência com TTA
===============================================

Inclui:
- Test-Time Augmentation (TTA)
- Detecção de incerteza
- Ensemble de predições
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


class TestTimeAugmentation:
    """
    Test-Time Augmentation para melhorar robustez da predição.
    
    Aplica múltiplas transformações na imagem de teste e faz
    ensemble das predições.
    """
    
    # Transformações padrão para TTA
    DEFAULT_TRANSFORMS = [
        # Original
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Center crop ligeiramente maior
        transforms.Compose([
            transforms.Resize((420, 420)),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Rotação leve
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Brilho/contraste sutil
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]
    
    def __init__(self, transforms_list: Optional[List[transforms.Compose]] = None):
        self.transforms = transforms_list or self.DEFAULT_TRANSFORMS
    
    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """
        Aplica todas as transformações TTA na imagem.
        
        Args:
            image: Imagem PIL
            
        Returns:
            Lista de tensores transformados
        """
        return [t(image) for t in self.transforms]
    
    @property
    def num_augmentations(self) -> int:
        return len(self.transforms)


class UncertaintyDetector:
    """
    Detecta incerteza nas predições do modelo.
    
    Usa múltiplas métricas:
    - Entropia da distribuição
    - Margem entre top-1 e top-2
    - Variância entre TTA predictions
    """
    
    def __init__(
        self,
        entropy_threshold: float = 0.8,
        margin_threshold: float = 0.2,
        tta_variance_threshold: float = 0.15,
    ):
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.tta_variance_threshold = tta_variance_threshold
    
    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calcula entropia da distribuição de probabilidades.
        
        Entropia alta = distribuição uniforme (incerto)
        Entropia baixa = distribuição picada (confiante)
        """
        # Evita log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs))
        # Normaliza pela entropia máxima
        max_entropy = np.log2(len(probabilities))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def compute_margin(self, probabilities: np.ndarray) -> float:
        """
        Calcula margem entre top-1 e top-2.
        
        Margem pequena = competição acirrada (incerto)
        Margem grande = predição clara (confiante)
        """
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) < 2:
            return 1.0
        return sorted_probs[0] - sorted_probs[1]
    
    def compute_tta_variance(
        self, 
        predictions_list: List[np.ndarray]
    ) -> Tuple[float, np.ndarray]:
        """
        Calcula variância entre predições TTA.
        
        Returns:
            (variância média, predição média)
        """
        stacked = np.stack(predictions_list)
        mean_pred = stacked.mean(axis=0)
        variance = stacked.var(axis=0).mean()
        return variance, mean_pred
    
    def is_uncertain(
        self,
        probabilities: np.ndarray,
        tta_predictions: Optional[List[np.ndarray]] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Determina se a predição é incerta.
        
        Returns:
            (is_uncertain, metrics_dict)
        """
        metrics = {}
        
        # Entropia
        entropy = self.compute_entropy(probabilities)
        metrics['entropy'] = entropy
        metrics['entropy_uncertain'] = entropy > self.entropy_threshold
        
        # Margem
        margin = self.compute_margin(probabilities)
        metrics['margin'] = margin
        metrics['margin_uncertain'] = margin < self.margin_threshold
        
        # TTA variance
        if tta_predictions is not None and len(tta_predictions) > 1:
            tta_var, mean_pred = self.compute_tta_variance(tta_predictions)
            metrics['tta_variance'] = tta_var
            metrics['tta_uncertain'] = tta_var > self.tta_variance_threshold
        else:
            metrics['tta_variance'] = 0.0
            metrics['tta_uncertain'] = False
        
        # Decisão final
        is_uncertain = (
            metrics['entropy_uncertain'] or
            metrics['margin_uncertain'] or
            metrics.get('tta_uncertain', False)
        )
        
        return is_uncertain, metrics


class AtlasLeafInference:
    """
    Pipeline completo de inferência AtlasLeaf v3.1.
    
    Suporta:
    - Inferência simples
    - Inferência com TTA
    - Detecção de incerteza
    - Retorno de detalhes completos
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_tta: bool = True,
        num_classes: int = 15,
        leaf_crop: bool = False,
        leaf_crop_mask_bg: bool = False,
    ):
        self.model = model
        self.device = device
        self.use_tta = use_tta
        self.num_classes = num_classes

        self.model.eval()
        self.model.to(device)

        self.tta = TestTimeAugmentation() if use_tta else None
        self.uncertainty_detector = UncertaintyDetector()

        # IMPORTANTE: o mesmo recorte usado no treino DEVE ser aplicado aqui.
        self.cropper = None
        if leaf_crop:
            from .leaf_segmentation import LeafCropper
            self.cropper = LeafCropper(mask_background=leaf_crop_mask_bg)
    
    def predict(
        self,
        image: Union[Image.Image, torch.Tensor],
        return_details: bool = False,
    ) -> Dict:
        """
        Faz predição na imagem.
        
        Args:
            image: Imagem PIL ou tensor
            return_details: Se True, retorna métricas detalhadas
            
        Returns:
            Dict com predição e metadados
        """
        # Aplica o recorte de folha (mesmo do treino) antes de tudo
        if self.cropper is not None and isinstance(image, Image.Image):
            image = self.cropper(image)

        with torch.no_grad():
            if self.use_tta and isinstance(image, Image.Image):
                return self._predict_with_tta(image, return_details)
            else:
                return self._predict_single(image, return_details)
    
    def _predict_single(
        self,
        image: Union[Image.Image, torch.Tensor],
        return_details: bool,
    ) -> Dict:
        """Predição simples sem TTA."""
        
        # Preprocessa se necessário
        if isinstance(image, Image.Image):
            transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            tensor = transform(image).unsqueeze(0).to(self.device)
        else:
            tensor = image.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        
        # Forward
        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1)
        
        # Converte para numpy
        probabilities = probs[0].cpu().numpy()
        predicted_class = int(probabilities.argmax())
        confidence = float(probabilities[predicted_class])
        
        # Monta resultado
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities,
            'is_uncertain': confidence < 0.7,
            'recommendation': self._get_recommendation(confidence),
        }
        
        if return_details:
            is_uncertain, metrics = self.uncertainty_detector.is_uncertain(probabilities)
            result['is_uncertain'] = is_uncertain
            result['uncertainty_metrics'] = metrics
        
        return result
    
    def _predict_with_tta(
        self,
        image: Image.Image,
        return_details: bool,
    ) -> Dict:
        """Predição com Test-Time Augmentation."""
        
        # Aplica TTA
        augmented_images = self.tta(image)
        
        # Coleta predições
        all_predictions = []
        all_logits = []
        
        for aug_img in augmented_images:
            tensor = aug_img.unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            
            all_predictions.append(probs[0].cpu().numpy())
            all_logits.append(logits[0].cpu().numpy())
        
        # Média das predições (ensemble)
        mean_probs = np.mean(all_predictions, axis=0)
        predicted_class = int(mean_probs.argmax())
        confidence = float(mean_probs[predicted_class])
        
        # Calcula variância entre TTA
        tta_variance = float(np.var([p[predicted_class] for p in all_predictions]))
        
        # Detecção de incerteza
        is_uncertain, metrics = self.uncertainty_detector.is_uncertain(
            mean_probs, all_predictions
        )
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': mean_probs,
            'is_uncertain': is_uncertain,
            'recommendation': self._get_recommendation(confidence, is_uncertain),
            'tta_variance': tta_variance,
            'num_tta': len(augmented_images),
        }
        
        if return_details:
            result['uncertainty_metrics'] = metrics
            result['individual_predictions'] = all_predictions
        
        return result
    
    def _get_recommendation(self, confidence: float, is_uncertain: bool = False) -> str:
        """Gera recomendação baseada na confiança."""
        if is_uncertain or confidence < 0.5:
            return "⚠️ BAIXA CONFIANÇA - Verifique visualmente ou capture nova imagem"
        elif confidence < 0.7:
            return "💡 CONFIANÇA MODERADA - Considere verificação adicional"
        elif confidence < 0.9:
            return "✅ BOA CONFIANÇA - Predição confiável"
        else:
            return "🎯 ALTA CONFIANÇA - Predição muito confiável"
    
    def predict_batch(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Faz predição em batch de imagens.
        
        Args:
            images: Lista de imagens
            batch_size: Tamanho do batch
            
        Returns:
            Lista de resultados
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                results.append(self.predict(img, return_details=False))
        return results


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def load_model_for_inference(
    checkpoint_path: str,
    model_name: str = "efficientnet_v2_s",
    num_classes: int = 15,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Carrega modelo treinado para inferência.
    
    Args:
        checkpoint_path: Caminho para o checkpoint .pth
        model_name: Nome do modelo
        num_classes: Número de classes
        device: Device para inferência
        
    Returns:
        (modelo, device)
    """
    from .model_v31 import create_model
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cria modelo
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,  # Carrega do checkpoint
    )
    
    # Carrega pesos
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, device


# =============================================================================
# TESTE
# =============================================================================

if __name__ == "__main__":
    print("Testando pipeline de inferência...")
    
    # Cria modelo dummy para teste
    from .model_v31 import create_model
    
    model = create_model("efficientnet_v2_s", num_classes=15)
    device = torch.device("cpu")
    
    # Cria pipeline
    pipeline = AtlasLeafInference(model, device, use_tta=True)
    
    # Cria imagem dummy
    dummy_img = Image.new('RGB', (512, 512), color='green')
    
    # Testa predição com TTA
    result = pipeline.predict(dummy_img, return_details=True)
    
    print(f"✅ Predição: Classe {result['predicted_class']}")
    print(f"✅ Confiança: {result['confidence']:.3f}")
    print(f"✅ Incerto: {result['is_uncertain']}")
    print(f"✅ TTA usado: {result.get('num_tta', 0)} augmentations")
