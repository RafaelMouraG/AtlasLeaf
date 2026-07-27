"""
AtlasLeaf v3.1 - Augmentations Avançadas
========================================

Inclui:
- Domain Randomization para simular condições de campo
- Augmentação adaptativa baseada na classe
"""

import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Optional


class DomainRandomization:
    """
    Simula variações de condições de campo em imagens de folhas.
    
    Inclui:
    - Sombras (simula nuvens, outros objetos)
    - Luz solar (overexposure)
    - Reflexos
    - Variação de saturação (folhas secas vs verdes)
    """
    
    def __init__(
        self,
        p: float = 0.5,
        shadow_p: float = 0.3,
        sunlight_p: float = 0.2,
        reflection_p: float = 0.1,
        saturation_p: float = 0.2,
    ):
        """
        Args:
            p: Probabilidade de aplicar qualquer augmentation
            shadow_p: Probabilidade de adicionar sombras
            sunlight_p: Probabilidade de simular luz solar
            reflection_p: Probabilidade de adicionar reflexos
            saturation_p: Probabilidade de variar saturação
        """
        self.p = p
        self.shadow_p = shadow_p
        self.sunlight_p = sunlight_p
        self.reflection_p = reflection_p
        self.saturation_p = saturation_p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Aplica domain randomization na imagem."""
        if random.random() > self.p:
            return image
        
        # Aplica transformações aleatórias
        if random.random() < self.shadow_p:
            image = self.add_shadow(image)
        
        if random.random() < self.sunlight_p:
            image = self.add_sunlight(image)
        
        if random.random() < self.reflection_p:
            image = self.add_reflection(image)
        
        if random.random() < self.saturation_p:
            image = self.vary_saturation(image)
        
        return image
    
    def add_shadow(self, image: Image.Image) -> Image.Image:
        """
        Adiciona sombra aleatória na imagem.
        Simula nuvens ou objetos próximos bloqueando luz.
        """
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Cria máscara de sombra
        shadow = np.ones((h, w), dtype=np.float32)
        
        # Tipo de sombra aleatório
        shadow_type = random.choice(['gradient', 'spot', 'band'])
        
        if shadow_type == 'gradient':
            # Gradiente diagonal
            direction = random.choice(['tl', 'tr', 'bl', 'br'])
            intensity = random.uniform(0.3, 0.7)
            
            for i in range(h):
                for j in range(w):
                    if direction == 'tl':
                        factor = 1 - (i + j) / (h + w) * intensity
                    elif direction == 'tr':
                        factor = 1 - (i + (w - j)) / (h + w) * intensity
                    elif direction == 'bl':
                        factor = 1 - ((h - i) + j) / (h + w) * intensity
                    else:  # br
                        factor = 1 - ((h - i) + (w - j)) / (h + w) * intensity
                    shadow[i, j] = max(0.3, factor)
        
        elif shadow_type == 'spot':
            # Mancha circular de sombra
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            radius = random.randint(min(h, w) // 4, min(h, w) // 2)
            intensity = random.uniform(0.4, 0.8)
            
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            mask = dist_from_center <= radius
            shadow[mask] = 1 - intensity * (1 - dist_from_center[mask] / radius)
        
        else:  # band
            # Faixa horizontal ou vertical
            if random.random() < 0.5:
                # Horizontal
                y_start = random.randint(0, h - h // 4)
                y_end = y_start + random.randint(h // 8, h // 4)
                shadow[y_start:y_end, :] *= random.uniform(0.4, 0.7)
            else:
                # Vertical
                x_start = random.randint(0, w - w // 4)
                x_end = x_start + random.randint(w // 8, w // 4)
                shadow[:, x_start:x_end] *= random.uniform(0.4, 0.7)
        
        # Aplica sombra
        for c in range(3):
            img_array[:, :, c] *= shadow
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def add_sunlight(self, image: Image.Image) -> Image.Image:
        """
        Simula efeito de luz solar forte (overexposure local).
        """
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(1.2, 1.5)
        return enhancer.enhance(factor)
    
    def add_reflection(self, image: Image.Image) -> Image.Image:
        """
        Adiciona reflexo sutil simulando luz em folha brilhosa.
        """
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Cria reflexo em posição aleatória
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(10, 30)
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        mask = dist <= radius
        intensity = random.uniform(1.1, 1.3)
        
        for c in range(3):
            channel = img_array[:, :, c]
            channel[mask] = np.clip(channel[mask] * intensity, 0, 255)
            img_array[:, :, c] = channel
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def vary_saturation(self, image: Image.Image) -> Image.Image:
        """
        Varia saturação para simular folhas em diferentes estágios.
        """
        enhancer = ImageEnhance.Color(image)
        # Pode diminuir (folha seca) ou aumentar (folha saudável)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)


class AdaptiveAugmentation:
    """
    Augmentação adaptativa baseada na frequência da classe.
    
    Classes raras recebem augmentação mais forte.
    """
    
    def __init__(self, class_count: int, intensity: str = "medium"):
        """
        Args:
            class_count: Número de amostras da classe
            intensity: 'light', 'medium', 'strong' ou 'auto'
        """
        self.class_count = class_count
        
        if intensity == "auto":
            # Determina automaticamente baseado na contagem
            if class_count < 50:
                self.intensity = "strong"
            elif class_count < 200:
                self.intensity = "medium"
            else:
                self.intensity = "light"
        else:
            self.intensity = intensity
    
    def get_transform_params(self) -> dict:
        """Retorna parâmetros de transformação baseado na intensidade."""
        params = {
            "light": {
                "rotation": 10,
                "scale": (0.9, 1.1),
                "flip_h": 0.5,
                "color_jitter": 0.1,
            },
            "medium": {
                "rotation": 25,
                "scale": (0.8, 1.2),
                "flip_h": 0.5,
                "flip_v": 0.3,
                "color_jitter": 0.2,
            },
            "strong": {
                "rotation": 45,
                "scale": (0.7, 1.3),
                "flip_h": 0.5,
                "flip_v": 0.5,
                "color_jitter": 0.3,
                "blur": (0.1, 1.5),
            }
        }
        return params.get(self.intensity, params["medium"])


class RandomErasing:
    """
    Random Erasing para simular oclusões (degradês, insetos, etc).
    
    Similar ao RandomErasing do torchvision mas para PIL Images.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Optional[float] = None,  # None = random noise
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        area = h * w
        
        for _ in range(10):  # Tentativas
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            eh = int(round(np.sqrt(target_area / aspect_ratio)))
            ew = int(round(np.sqrt(target_area * aspect_ratio)))
            
            if eh < h and ew < w:
                x1 = random.randint(0, w - ew)
                y1 = random.randint(0, h - eh)
                
                if self.value is None:
                    # Preenche com ruído aleatório
                    img_array[y1:y1+eh, x1:x1+ew] = np.random.randint(
                        0, 256, (eh, ew, 3), dtype=np.uint8
                    )
                else:
                    # Preenche com valor fixo
                    img_array[y1:y1+eh, x1:x1+ew] = int(self.value * 255)
                
                break
        
        return Image.fromarray(img_array)


# =============================================================================
# TESTE
# =============================================================================

if __name__ == "__main__":
    print("Testando augmentations...")
    
    # Cria imagem de teste
    test_img = Image.new('RGB', (384, 384), color=(100, 150, 50))
    
    # Testa DomainRandomization
    dr = DomainRandomization(p=1.0)
    aug_img = dr(test_img)
    print(f"✅ DomainRandomization aplicada")
    
    # Testa AdaptiveAugmentation
    aa = AdaptiveAugmentation(class_count=30, intensity="auto")
    params = aa.get_transform_params()
    print(f"✅ AdaptiveAugmentation (classe rara): {aa.intensity}")
    print(f"   Params: {params}")
    
    aa_common = AdaptiveAugmentation(class_count=500, intensity="auto")
    print(f"✅ AdaptiveAugmentation (classe comum): {aa_common.intensity}")
    
    # Testa RandomErasing
    re = RandomErasing(p=1.0)
    erased = re(test_img)
    print(f"✅ RandomErasing aplicada")
