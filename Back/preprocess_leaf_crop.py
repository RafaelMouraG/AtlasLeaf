"""
AtlasLeaf - Pré-processamento: recorte de folha em lote
=======================================================

Lê datasets/unified/ e gera datasets/unified_cropped/ com a mesma estrutura,
aplicando LeafCropper a cada imagem. Copia manifest.json e splits*.json
(os caminhos relativos são idênticos, então continuam válidos).

Uso:
    python preprocess_leaf_crop.py                 # bbox crop (seguro)
    python preprocess_leaf_crop.py --mask-bg       # + pinta fundo de cinza
    python preprocess_leaf_crop.py --src datasets/unified --dst datasets/unified_cropped
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline.leaf_segmentation import LeafCropper

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/unified")
    ap.add_argument("--dst", default="datasets/unified_cropped")
    ap.add_argument("--mask-bg", action="store_true", help="Pinta o fundo de cinza (mais agressivo)")
    ap.add_argument("--no-crop", action="store_true",
                    help="NÃO recorta a folha; só reduz a resolução (acelera o treino sem o recorte que piorava)")
    ap.add_argument("--max-side", type=int, default=768, help="Reduz o lado maior da imagem recortada (economia de disco/IO)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    cropper = LeafCropper(mask_background=args.mask_bg)

    # Copia manifest/splits (caminhos relativos permanecem válidos)
    for meta in ["manifest.json", "splits.json", "splits_source.json", "splits_camera.json", "errors.log"]:
        p = src / meta
        if p.exists():
            shutil.copy2(p, dst / meta)

    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Encontradas {len(imgs)} imagens. mask_bg={args.mask_bg} -> {dst}")

    t0 = time.time()
    done = fail = skipped = 0
    for i, p in enumerate(imgs):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            im = Image.open(p).convert("RGB")
            cr = im if args.no_crop else cropper(im)
            if args.max_side and max(cr.size) > args.max_side:
                cr.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
            # salva sempre como jpg de qualidade alta
            out = out.with_suffix(".jpg")
            cr.save(out, "JPEG", quality=92)
            done += 1
        except Exception as e:
            fail += 1
            print(f"  [falha] {rel}: {e}")
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            rate = (done + fail) / el if el else 0
            print(f"  {i+1}/{len(imgs)}  ok={done} skip={skipped} fail={fail}  "
                  f"{rate:.1f} img/s  eta {(len(imgs)-i-1)/max(rate,1e-6):.0f}s", flush=True)

    print(f"\nConcluído: ok={done} skip={skipped} fail={fail} em {time.time()-t0:.0f}s")
    print(f"Dataset recortado em: {dst}")


if __name__ == "__main__":
    main()
