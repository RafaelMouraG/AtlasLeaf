"""Constrói o dataset unificado 768px (ASDID + Mendeley V2) e splits honestos.

Lê as imagens diretamente dos arquivos ZIP verificados em ``datasets/raw/``,
extrai o EXIF original (câmera Canon vs. Moto) ANTES de redimensionar, salva
JPEGs em 768px em ``datasets/unified_resized/`` e gera três splits compatíveis
com ``scripts/train_field7.py --source-split``:

1. ``splits_camera.json`` — baseline canônico (treino/val Canon, teste Moto).
2. ``splits_camera_mendeley_train.json`` — Canon + Mendeley V2 no treino/val,
   mesmo holdout Moto para comparação 1-para-1.
3. ``splits_mendeley_holdout.json`` — treino/val ASDID (Canon + Moto), teste
   exclusivo no Mendeley V2 (avaliação cross-região / cross-fonte).

Opcionalmente empacota tudo em um ``.tar.gz`` pronto para subir ao Google Drive
e treinar no Google Colab Pro.

Uso:
    python scripts/build_colab_dataset.py --archive datasets/atlasleaf_unified_resized_768.tar.gz
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

BACK_DIR = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELD7_CLASS_IDS = [0, 1, 2, 3, 7, 12, 13]

CLASS_CATALOG = {
    "healthy": (0, "Saudável"),
    "asian_rust": (1, "Ferrugem Asiática"),
    "target_spot": (2, "Mancha Alvo"),
    "cercospora_blight": (3, "Crestamento de Cercospora"),
    "bacterial_blight": (5, "Mancha Bacteriana"),
    "downy_mildew": (7, "Míldio"),
    "sudden_death_syndrome": (10, "Síndrome da Morte Súbita"),
    "potassium_deficiency": (12, "Deficiência de Potássio"),
    "frogeye_leaf_spot": (13, "Mancha Olho-de-Rã"),
}

ASDID_ZIP_TO_CLASS = {
    "healthy.zip": "healthy",
    "soybean_rust.zip": "asian_rust",
    "target_spot.zip": "target_spot",
    "cercospora_leaf_blight.zip": "cercospora_blight",
    "bacterial_blight.zip": "bacterial_blight",
    "downey_mildew.zip": "downy_mildew",
    "potassium_deficiency.zip": "potassium_deficiency",
    "frogeye.zip": "frogeye_leaf_spot",
}

MENDELEY_FOLDER_TO_CLASS = {
    "healthy": "healthy",
    "rust": "asian_rust",
    "cercospora leaf blight": "cercospora_blight",
    "bacterial blight": "bacterial_blight",
    "sudden death syndrome": "sudden_death_syndrome",
}


def detect_camera(im: Image.Image) -> str:
    """Detecta câmera a partir do EXIF original (mesma regra de camera_split.py)."""
    try:
        ex = im._getexif() or {}
        model = str({TAGS.get(k, k): v for k, v in ex.items()}.get("Model", "")).lower()
        if "canon" in model:
            return "Canon"
        if "moto" in model:
            return "Moto"
        if model.strip():
            return model.strip()[:40]
    except Exception:
        pass
    return "?"


def process_zip_job(job: dict) -> list[dict]:
    """Processa um arquivo ZIP em paralelo, lendo EXIF antes de redimensionar."""
    zip_path = Path(job["zip_path"])
    out_dir = Path(job["out_dir"])
    source = job["source"]
    max_side = job["max_side"]
    quality = job["quality"]
    mode = job["mode"]

    records = []
    errors = []

    with zipfile.ZipFile(zip_path) as zf:
        entries = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = Path(info.filename)
            if p.name.startswith(".") or "__MACOSX" in p.parts:
                continue
            if p.suffix.lower() not in IMG_EXT:
                continue
            if mode == "asdid":
                cls_name = job["class_name"]
            else:
                folder = p.parent.name.strip().lower()
                cls_name = MENDELEY_FOLDER_TO_CLASS.get(folder)
                if cls_name is None:
                    continue
            entries.append((cls_name, info))

        entries.sort(key=lambda item: (item[0], item[1].filename))
        per_class_counter = Counter()

        for cls_name, info in entries:
            per_class_counter[cls_name] += 1
            idx = per_class_counter[cls_name]
            cid, _ = CLASS_CATALOG[cls_name]
            rel_path = f"{cls_name}/{source}_{idx:05d}.jpg"
            dst_path = out_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                raw = zf.read(info)
                with Image.open(io.BytesIO(raw)) as im:
                    camera = detect_camera(im) if source == "asdid" else "mendeley_v2"
                    orig_w, orig_h = im.size
                    rgb = im.convert("RGB")
                if max_side and max(rgb.size) > max_side:
                    rgb.thumbnail((max_side, max_side), Image.LANCZOS)
                rgb.save(dst_path, "JPEG", quality=quality)
                records.append({
                    "path": rel_path,
                    "class_id": cid,
                    "class_name": cls_name,
                    "source": source,
                    "camera": camera,
                    "orig_file": info.filename,
                    "orig_size": [orig_w, orig_h],
                })
            except Exception as exc:
                errors.append(f"{zip_path.name}:{info.filename}: {exc}")

    return {"zip": zip_path.name, "records": records, "errors": errors}


def build_splits(images: list[dict], class_names: dict[int, str], val_frac: float = 0.15, seed: int = 42):
    """Gera os três splits honestos preservando holdouts sem vazamento."""
    # --- 1. splits_camera.json (baseline canônico: Canon -> Moto) ---
    rng = random.Random(seed)
    by_cls_cam = defaultdict(lambda: defaultdict(list))
    by_cls_mendeley = defaultdict(list)

    for im in images:
        cid = int(im["class_id"])
        if cid not in FIELD7_CLASS_IDS:
            continue
        if im["source"] == "asdid" and im["camera"] in ("Canon", "Moto"):
            by_cls_cam[cid][im["camera"]].append(im["path"])
        elif im["source"] == "mendeleyv2":
            by_cls_mendeley[cid].append(im["path"])

    cam_train, cam_val, cam_test, cam_report = [], [], [], []
    for cid in FIELD7_CLASS_IDS:
        canon = sorted(by_cls_cam[cid]["Canon"])
        moto = sorted(by_cls_cam[cid]["Moto"])
        rng.shuffle(canon)
        n_val = max(1, int(len(canon) * val_frac)) if canon else 0
        cam_val.extend(canon[:n_val])
        cam_train.extend(canon[n_val:])
        cam_test.extend(moto)
        cam_report.append({
            "class_id": cid,
            "class": class_names[cid],
            "canon": len(canon),
            "moto_test": len(moto),
            "mendeleyv2": len(by_cls_mendeley[cid]),
        })

    splits_camera = {
        "_meta": {
            "strategy": "camera-holdout",
            "train_cam": "Canon",
            "test_cam": "Moto",
            "field_classes": FIELD7_CLASS_IDS,
            "seed": seed,
            "note": "test = câmera Moto (não vista). Baseline honesto campo→campo.",
        },
        "train": sorted(set(cam_train)),
        "val": sorted(set(cam_val)),
        "test": sorted(set(cam_test)),
        "test_insource": [],
        "report": cam_report,
    }

    # --- 2. splits_camera_mendeley_train.json (Canon + Mendeley V2 -> Moto) ---
    rng_m = random.Random(seed)
    cm_train = list(cam_train)
    cm_val = list(cam_val)
    for cid in FIELD7_CLASS_IDS:
        mend = sorted(by_cls_mendeley[cid])
        if not mend:
            continue
        rng_m.shuffle(mend)
        n_val = max(1, int(len(mend) * val_frac))
        cm_val.extend(mend[:n_val])
        cm_train.extend(mend[n_val:])

    splits_camera_mendeley = {
        "_meta": {
            "strategy": "camera-holdout-plus-mendeley",
            "train_sources": ["asdid:Canon", "mendeleyv2"],
            "test_cam": "Moto",
            "field_classes": FIELD7_CLASS_IDS,
            "seed": seed,
            "note": "Adiciona Mendeley V2 ao treino/val mantendo EXATAMENTE o mesmo holdout Moto para comparação justa.",
        },
        "train": sorted(set(cm_train)),
        "val": sorted(set(cm_val)),
        "test": sorted(set(cam_test)),
        "test_insource": [],
        "report": cam_report,
    }

    # --- 3. splits_mendeley_holdout.json (ASDID inteiro -> Mendeley V2) ---
    rng_h = random.Random(seed)
    mh_train, mh_val, mh_test = [], [], []
    for cid in FIELD7_CLASS_IDS:
        asdid_all = sorted(by_cls_cam[cid]["Canon"] + by_cls_cam[cid]["Moto"])
        rng_h.shuffle(asdid_all)
        n_val = max(1, int(len(asdid_all) * val_frac)) if asdid_all else 0
        mh_val.extend(asdid_all[:n_val])
        mh_train.extend(asdid_all[n_val:])
        mh_test.extend(sorted(by_cls_mendeley[cid]))

    splits_mendeley_holdout = {
        "_meta": {
            "strategy": "cross-dataset-mendeley-holdout",
            "train_sources": ["asdid:Canon", "asdid:Moto"],
            "test_source": "mendeleyv2",
            "field_classes": FIELD7_CLASS_IDS,
            "seed": seed,
            "note": "Teste cross-região (Mendeley V2 fora do treino; cobre healthy, asian_rust e cercospora_blight).",
        },
        "train": sorted(set(mh_train)),
        "val": sorted(set(mh_val)),
        "test": sorted(set(mh_test)),
        "test_insource": [],
        "report": cam_report,
    }

    return splits_camera, splits_camera_mendeley, splits_mendeley_holdout


def create_archive(dataset_dir: Path, archive_path: Path):
    """Empacota dataset_dir em um .tar.gz leve para subir ao Google Drive."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with tarfile.open(tmp_path, "w:gz", compresslevel=6) as tar:
        tar.add(dataset_dir, arcname=dataset_dir.name)
    tmp_path.replace(archive_path)


def main():
    ap = argparse.ArgumentParser(description="Prepara dataset 768px + splits honestos para Colab Pro")
    ap.add_argument("--asdid-raw", default=str(BACK_DIR / "datasets/raw/asdid"))
    ap.add_argument("--mendeley-zip", default=str(BACK_DIR / "datasets/raw/mendeley_v2/mendeley_v2.zip"))
    ap.add_argument("--out", default=str(BACK_DIR / "datasets/unified_resized"))
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--archive", default=None, help="Opcional: caminho do .tar.gz final para Google Drive")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asdid_raw = Path(args.asdid_raw)
    mendeley_zip = Path(args.mendeley_zip)

    jobs = []
    for zip_name, cls_name in sorted(ASDID_ZIP_TO_CLASS.items()):
        zp = asdid_raw / zip_name
        if zp.exists():
            jobs.append({
                "zip_path": str(zp),
                "out_dir": str(out_dir),
                "source": "asdid",
                "mode": "asdid",
                "class_name": cls_name,
                "max_side": args.max_side,
                "quality": args.quality,
            })
        else:
            print(f"⚠️ ASDID zip ausente (pulando): {zp}")

    if mendeley_zip.exists():
        jobs.append({
            "zip_path": str(mendeley_zip),
            "out_dir": str(out_dir),
            "source": "mendeleyv2",
            "mode": "mendeley",
            "class_name": None,
            "max_side": args.max_side,
            "quality": args.quality,
        })
    else:
        print(f"⚠️ Mendeley V2 zip ausente (pulando): {mendeley_zip}")

    if not jobs:
        raise FileNotFoundError("Nenhum ZIP de dataset encontrado.")

    print(f"🚀 Processando {len(jobs)} arquivos ZIP com {args.workers} workers (max_side={args.max_side}px)...")
    all_records = []
    all_errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_zip_job, j): j["zip_path"] for j in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            all_records.extend(res["records"])
            all_errors.extend(res["errors"])
            print(f"  ✅ {res['zip']:30s} -> {len(res['records']):5d} imgs ({len(res['errors'])} erros)", flush=True)

    all_records.sort(key=lambda r: r["path"])

    # Monta manifest.json no contrato canônico do AtlasLeaf
    dist = {}
    class_names = {}
    for cls_name, (cid, friendly) in CLASS_CATALOG.items():
        class_names[cid] = cls_name
    for rec in all_records:
        nm = rec["class_name"]
        cid, friendly = CLASS_CATALOG[nm]
        dist.setdefault(nm, {"id": cid, "count": 0, "friendly_name": friendly})
        dist[nm]["count"] += 1

    manifest = {
        "version": "colab-768-1.0",
        "max_side": args.max_side,
        "total_images": len(all_records),
        "datasets_used": sorted({r["source"] for r in all_records}),
        "class_distribution": dist,
        "images": [
            {
                "path": r["path"],
                "class_id": r["class_id"],
                "source": r["source"],
                "camera": r["camera"],
            }
            for r in all_records
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if all_errors:
        (out_dir / "errors.log").write_text("\n".join(all_errors) + "\n", encoding="utf-8")

    # Gera splits honestos
    s_cam, s_cam_mend, s_mend_hold = build_splits(all_records, class_names)
    (out_dir / "splits_camera.json").write_text(
        json.dumps(s_cam, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "splits_camera_mendeley_train.json").write_text(
        json.dumps(s_cam_mend, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "splits_mendeley_holdout.json").write_text(
        json.dumps(s_mend_hold, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Relatório detalhado
    cam_counts = Counter((r["source"], r["camera"]) for r in all_records)
    print(f"\n{'='*72}")
    print(f"📊 DATASET PRONTO EM {out_dir} ({time.time()-t0:.1f}s)")
    print(f"{'='*72}")
    print(f"Total de imagens: {len(all_records)} (erros: {len(all_errors)})")
    print("Fontes / câmeras detectadas via EXIF original:")
    for (src, cam), cnt in sorted(cam_counts.items()):
        print(f"  - {src:12s} | {cam:16s}: {cnt:5d}")

    print("\nClasses field7 nos splits:")
    print(f"{'cid':>3} | {'classe':22s} | {'Canon':>6} | {'Moto(test)':>10} | {'MendeleyV2':>10}")
    print("-" * 64)
    for r in s_cam["report"]:
        print(f"{r['class_id']:>3} | {r['class']:22s} | {r['canon']:>6} | {r['moto_test']:>10} | {r['mendeleyv2']:>10}")

    print("\nResumo dos 3 splits:")
    print(f"  1. splits_camera.json                : train={len(s_cam['train'])} val={len(s_cam['val'])} test(Moto)={len(s_cam['test'])}")
    print(f"  2. splits_camera_mendeley_train.json : train={len(s_cam_mend['train'])} val={len(s_cam_mend['val'])} test(Moto)={len(s_cam_mend['test'])}")
    print(f"  3. splits_mendeley_holdout.json      : train={len(s_mend_hold['train'])} val={len(s_mend_hold['val'])} test(Mendeley)={len(s_mend_hold['test'])}")

    if args.archive:
        arch = Path(args.archive)
        print(f"\n📦 Empacotando {arch} ...")
        t1 = time.time()
        create_archive(out_dir, arch)
        size_mb = arch.stat().st_size / (1024 * 1024)
        print(f"✅ Arquivo pronto para o Google Drive: {arch} ({size_mb:.1f} MB em {time.time()-t1:.1f}s)")


if __name__ == "__main__":
    main()
