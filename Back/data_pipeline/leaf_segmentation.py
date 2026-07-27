"""
AtlasLeaf - Segmentação e Recorte de Folha
==========================================

Objetivo: reduzir o viés de fonte (fundo, cena, enquadramento) isolando a folha
antes do treino/inferência. Sem dependências além de numpy/scipy/PIL.

Método (clássico, dependency-free):
    1. Índice de vegetação ExG = 2g - r - b (coordenadas cromáticas normalizadas)
       -> realça tecido de folha (verde E amarelado), suprime solo/mão/fundo.
    2. Threshold de Otsu -> máscara binária.
    3. Limpeza morfológica + preenchimento de buracos + maior componente conectado.
    4. Se a máscara é "confiável" (fração de foreground em faixa razoável):
         recorta pela bounding box com margem; opcionalmente pinta o fundo de cinza.
       Senão: fallback para center-crop (nunca destrói a imagem).

IMPORTANTE: o MESMO recorte deve ser aplicado no treino E na inferência.
Use `LeafCropper` como primeira etapa do pipeline de transforms (antes do Resize).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def _otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Threshold de Otsu para um array 1D de valores contínuos."""
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin
    hist, edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return (vmin + vmax) / 2.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = np.cumsum(hist)
    w_back = w / total
    w_fore = 1.0 - w_back
    cum_mean = np.cumsum(hist * centers) / total
    global_mean = cum_mean[-1]
    denom = w_back * w_fore
    denom[denom == 0] = 1e-12
    between = (global_mean * w_back - cum_mean) ** 2 / denom
    idx = int(np.argmax(between))
    return float(centers[idx])


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Calcula máscara binária da folha via ExG + Otsu + maior componente.

    Args:
        rgb: array HxWx3 uint8
    Returns:
        máscara booleana HxW (True = folha)
    """
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    s = r + g + b + 1e-6
    rn, gn, bn = r / s, g / s, b / s
    exg = 2.0 * gn - rn - bn  # faixa ~[-1, 2]

    thr = _otsu_threshold(exg)
    # Garante que o threshold não seja degenerado (folha deve ser o lado "verde")
    mask = exg > thr

    if mask.mean() < 0.5:
        # ok: foreground é a minoria verde
        pass
    else:
        # Otsu pode ter invertido em imagens quase-toda-folha; mantém como está.
        pass

    # Limpeza morfológica
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)

    # Maior componente conectado
    labeled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
        biggest = int(np.argmax(sizes)) + 1
        mask = labeled == biggest
    elif n == 0:
        mask = np.zeros(rgb.shape[:2], dtype=bool)

    return mask


class LeafCropper:
    """
    Transform PIL->PIL que isola e recorta a folha.

    Args:
        margin: margem relativa em volta da bounding box (0.12 = 12%).
        mask_background: se True, pinta o fundo (fora da folha) de cinza neutro.
        bg_color: cor do fundo quando mask_background=True.
        min_fg / max_fg: faixa de fração de foreground considerada "confiável".
                         Fora dela cai no fallback de center-crop.
        center_crop_frac: fração central usada no fallback.
        work_size: lado máximo para computar a máscara (velocidade); a bbox é
                   reescalada para a resolução original.
    """

    def __init__(
        self,
        margin: float = 0.12,
        mask_background: bool = False,
        bg_color: tuple = (124, 116, 104),
        min_fg: float = 0.02,
        max_fg: float = 0.95,
        center_crop_frac: float = 0.85,
        work_size: int = 384,
    ):
        self.margin = margin
        self.mask_background = mask_background
        self.bg_color = bg_color
        self.min_fg = min_fg
        self.max_fg = max_fg
        self.center_crop_frac = center_crop_frac
        self.work_size = work_size

    def _center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        cw, ch = int(w * self.center_crop_frac), int(h * self.center_crop_frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch))

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size

        # Downscale para calcular a máscara rápido
        scale = min(1.0, self.work_size / max(w, h))
        if scale < 1.0:
            small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        else:
            small = img
        rgb_small = np.asarray(small)

        try:
            mask = leaf_mask(rgb_small)
        except Exception:
            return self._center_crop(img)

        fg = mask.mean()
        if fg < self.min_fg or fg > self.max_fg:
            # Segmentação não confiável -> fallback seguro
            return self._center_crop(img)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return self._center_crop(img)

        # bbox na resolução pequena -> reescala para original
        inv = 1.0 / scale if scale < 1.0 else 1.0
        x0, x1 = xs.min() * inv, xs.max() * inv
        y0, y1 = ys.min() * inv, ys.max() * inv

        bw, bh = x1 - x0, y1 - y0
        x0 = max(0, int(x0 - bw * self.margin))
        y0 = max(0, int(y0 - bh * self.margin))
        x1 = min(w, int(x1 + bw * self.margin))
        y1 = min(h, int(y1 + bh * self.margin))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return self._center_crop(img)

        if self.mask_background:
            full_mask = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
            full_mask = np.asarray(full_mask) > 127
            out = np.asarray(img).copy()
            out[~full_mask] = np.array(self.bg_color, dtype=np.uint8)
            img = Image.fromarray(out)

        return img.crop((x0, y0, x1, y1))


if __name__ == "__main__":
    import sys
    cropper = LeafCropper(mask_background=False)
    if len(sys.argv) > 1:
        im = Image.open(sys.argv[1])
        out = cropper(im)
        print(f"in={im.size} -> out={out.size}")
        out.save("/tmp/leaf_crop_demo.png")
        print("salvo em /tmp/leaf_crop_demo.png")
    else:
        # smoke test
        im = Image.new("RGB", (400, 300), (120, 90, 60))
        im.paste((60, 140, 40), (120, 80, 280, 220))  # "folha" verde no centro
        print("bbox crop:", cropper(im).size)
