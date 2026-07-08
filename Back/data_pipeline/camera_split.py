"""
AtlasLeaf - Split por CÂMERA (avaliação campo→campo, relevante p/ deploy)
=========================================================================

O ASDID é campo real, mas com 2 câmeras (Canon EOS 7D Mark II e Motorola moto z4).
O teste honesto de generalização para OUTRA lavoura/telefone é: treinar numa câmera
e testar na outra. Restringe às classes de campo que têm AMBAS as câmeras.

Gera `splits_camera.json` (compatível com load_presplit_dataset / --source-split):
  - train/val  = imagens Canon (Canon é a maior)
  - test       = imagens Moto (câmera não vista) -> HONESTO campo→campo
  - test_insource = vazio (todas as classes são cross-camera)

Uso:
    python -m data_pipeline.camera_split --data-dir datasets/unified
Lê EXIF do dataset ORIGINAL (não recortado). Escreve o json no data-dir e, se existir,
também em datasets/unified_cropped (os caminhos relativos são idênticos).
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

# Classes de campo (ASDID) com AMBAS as câmeras — as únicas mensuráveis cross-camera.
FIELD_CLASSES = [0, 1, 2, 3, 7, 12, 13]  # healthy, asian_rust, target, cercospora, downy, potassium, frogeye


def camera_of(path: Path) -> str:
    try:
        ex = Image.open(path)._getexif() or {}
        mo = str({TAGS.get(k, k): v for k, v in ex.items()}.get("Model", "")).lower()
        if "canon" in mo:
            return "Canon"
        if "moto" in mo:
            return "Moto"
    except Exception:
        pass
    return "?"


def build(data_dir: Path, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    m = json.load(open(data_dir / "manifest.json"))
    names = {v["id"]: k for k, v in m["class_distribution"].items()}

    by_cls_cam = defaultdict(lambda: defaultdict(list))
    for img in m["images"]:
        cid = int(str(img["class_id"]))
        if cid not in FIELD_CLASSES:
            continue
        rp = img["path"]
        # só ASDID tem EXIF de câmera; ignora outras fontes
        if not rp.split("/")[-1].startswith("asdid"):
            continue
        cam = camera_of(data_dir / rp)
        if cam in ("Canon", "Moto"):
            by_cls_cam[cid][cam].append(rp)

    train, val, test, report = [], [], [], []
    for cid in FIELD_CLASSES:
        canon = list(by_cls_cam[cid].get("Canon", []))
        moto = list(by_cls_cam[cid].get("Moto", []))
        rng.shuffle(canon)
        n_val = max(1, int(len(canon) * val_frac))
        val.extend(canon[:n_val])
        train.extend(canon[n_val:])
        test.extend(moto)
        report.append({"class_id": cid, "class": names.get(cid, "?"),
                       "canon": len(canon), "moto_test": len(moto)})

    splits = {
        "_meta": {"strategy": "camera-holdout", "train_cam": "Canon", "test_cam": "Moto",
                  "field_classes": FIELD_CLASSES, "seed": seed,
                  "note": "test = câmera Moto (não vista). Proxy campo→campo p/ deploy."},
        "train": sorted(set(train)),
        "val": sorted(set(val)),
        "test": sorted(set(test)),
        "test_insource": [],
        "report": report,
    }
    return splits, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets/unified")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    splits, _ = build(data_dir)

    print(f"\n{'='*64}\nSPLIT POR CÂMERA (train=Canon, test=Moto) — campo→campo\n{'='*64}")
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  test(Moto)={len(splits['test'])}")
    print(f"\n{'cid':>3} | {'classe':22s} | Canon(tr+val) | Moto(test)")
    print("-" * 56)
    for r in splits["report"]:
        print(f"{r['class_id']:>3} | {r['class']:22s} | {r['canon']:11d} | {r['moto_test']}")

    outs = [data_dir / "splits_camera.json"]
    cropped = data_dir.parent / "unified_cropped"
    if cropped.exists():
        outs.append(cropped / "splits_camera.json")
    for o in outs:
        json.dump(splits, open(o, "w"), indent=2, ensure_ascii=False)
        print(f"Salvo: {o}")


if __name__ == "__main__":
    main()
