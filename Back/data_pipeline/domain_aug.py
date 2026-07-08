"""
AtlasLeaf - Augmentation de DOMÍNIO
===================================

Objetivo: destruir as "assinaturas de fonte" que vivem nas estatísticas de pixel
(nitidez/resolução nativa, compressão JPEG, ciência de cor da câmera). O linear
probe mostrou que essas pistas sobrevivem ao recorte de folha e entregam a fonte
com ~99% — então aqui atacamos elas diretamente, só no TREINO.

Todas as transforms operam em PIL.Image -> PIL.Image (antes de ToTensor).
"""
from __future__ import annotations

import io
import random
from PIL import Image


class RandomJPEG:
    """Recomprime a imagem em JPEG com qualidade aleatória.

    Faz o modelo ver artefatos de compressão variados, em vez de aprender o
    'nível de JPEG' característico de cada dataset.
    """
    def __init__(self, p: float = 0.5, quality=(30, 90)):
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(int(self.quality[0]), int(self.quality[1]))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomResolution:
    """Reduz e volta a ampliar (bilinear) para embaralhar a nitidez/resolução nativa.

    ASDID (4000px) vs Kaggle (250px) têm nitidez muito diferente; isso apaga essa
    pista sem mudar o conteúdo.
    """
    def __init__(self, p: float = 0.5, scale=(0.3, 1.0)):
        self.p = p
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.scale[0], self.scale[1])
        nw, nh = max(8, int(w * s)), max(8, int(h * s))
        small = img.resize((nw, nh), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def domain_transforms():
    """Lista de transforms PIL->PIL para randomização de domínio (treino).

    Colocar ANTES do Resize/ToTensor. O ColorJitter (cor da câmera) fica no
    pipeline torchvision principal — aqui cuidamos de resolução e compressão.
    """
    return [
        RandomResolution(p=0.5, scale=(0.3, 1.0)),
        RandomJPEG(p=0.5, quality=(30, 90)),
    ]


if __name__ == "__main__":
    im = Image.new("RGB", (400, 300), (80, 140, 60))
    for t in domain_transforms():
        im2 = t(im)
        print(type(t).__name__, "->", im2.size)
