"""
AtlasLeaf - Integração do dataset baiano com o ASDID
====================================================

Junta suas fotos (bahia_raw/{fazenda}/{doenca}/*.jpg) com o ASDID num dataset
combinado e gera um split por FAZENDA (segura uma fazenda inteira p/ teste) — o
proxy honesto de "funciona numa lavoura que eu não amostrei".

Não duplica o ASDID: usa symlinks. Doença nova (não existe no ASDID) ganha um id novo.

Uso:
    python scripts/integrate_dataset.py --bahia-dir bahia_raw
    python scripts/integrate_dataset.py --bahia-dir bahia_raw --test-farm fazenda_riogrande

Depois:
    python scripts/train_field7.py --data-dir datasets/combined --source-split \
        --split-file splits_region.json --freeze-backbone --domain-aug \
        --num-classes <N imprimido pelo script>
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Classes de campo do ASDID que entram por padrão (as 7 confiáveis).
ASDID_FIELD_CLASSES = ["healthy", "asian_rust", "target_spot", "cercospora_blight",
                       "downy_mildew", "potassium_deficiency", "frogeye_leaf_spot"]


def load_class_map(unified: Path):
    """name -> id a partir do manifest do ASDID; permite estender com ids novos."""
    man = json.load(open(unified / "manifest.json"))
    name2id, friendly = {}, {}
    for name, info in man["class_distribution"].items():
        name2id[name] = int(info["id"])
        friendly[name] = info.get("friendly_name", name)
    return name2id, friendly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bahia-dir", required=True, help="Estrutura {fazenda}/{doenca}/*.jpg")
    ap.add_argument("--asdid-dir", default="datasets/unified")
    ap.add_argument("--out", default="datasets/combined")
    ap.add_argument("--asdid-classes", nargs="*", default=ASDID_FIELD_CLASSES,
                    help="Classes do ASDID a incluir (default: as 7 de campo)")
    ap.add_argument("--test-farm", default=None,
                    help="Fazenda a segurar p/ teste. Default: a de menor volume (se houver ≥2).")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    unified = Path(args.asdid_dir).resolve()
    bahia = Path(args.bahia_dir).resolve()
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    name2id, friendly = load_class_map(unified)
    next_id = max(name2id.values()) + 1

    # ---- coleta imagens do ASDID (só classes escolhidas, só arquivos asdid_*) ----
    images = []  # dicts: path(rel), class_id, source, farm
    def link(src_abs: Path, rel: str):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src_abs, dst)

    for cls in args.asdid_classes:
        d = unified / cls
        if not d.exists():
            print(f"⚠️ classe ASDID ausente: {cls}"); continue
        cid = name2id[cls]
        for f in sorted(d.glob("*")):
            if f.suffix.lower() in IMG_EXT and f.name.startswith("asdid"):
                rel = f"{cls}/{f.name}"
                link(f.resolve(), rel)
                images.append({"path": rel, "class_id": cid, "source": "asdid", "farm": "asdid"})

    # ---- coleta imagens da Bahia ({fazenda}/{doenca}/*) ----
    bahia_counts = defaultdict(Counter)  # cls -> farm -> n
    if not bahia.exists():
        raise FileNotFoundError(f"Pasta baiana não encontrada: {bahia}")
    for farm_dir in sorted(p for p in bahia.iterdir() if p.is_dir()):
        farm = farm_dir.name
        if farm.lower() in ("descartar", "discard", "registro"):
            continue
        for cls_dir in sorted(p for p in farm_dir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            if cls == "descartar":
                continue
            if cls not in name2id:  # doença nova -> id novo
                name2id[cls] = next_id
                friendly[cls] = cls.replace("_", " ").title()
                print(f"🆕 doença nova '{cls}' -> id {next_id}")
                next_id += 1
            cid = name2id[cls]
            n = 0
            for f in sorted(cls_dir.rglob("*")):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                n += 1
                rel = f"{cls}/bahia-{farm}_{n:04d}{f.suffix.lower()}"
                link(f.resolve(), rel)
                images.append({"path": rel, "class_id": cid, "source": "bahia", "farm": farm})
            bahia_counts[cls][farm] += n

    if not any(im["source"] == "bahia" for im in images):
        print("⚠️ Nenhuma imagem baiana encontrada. Confira a estrutura {fazenda}/{doenca}/.")

    # ---- manifest combinado ----
    id2name = {v: k for k, v in name2id.items()}
    dist = {}
    for im in images:
        nm = id2name[im["class_id"]]
        dist.setdefault(nm, {"id": im["class_id"], "count": 0,
                             "friendly_name": friendly.get(nm, nm)})
        dist[nm]["count"] += 1
    manifest = {"version": "combined-1.0", "total_images": len(images),
                "datasets_used": ["asdid", "bahia"], "class_distribution": dist,
                "images": images}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2, ensure_ascii=False)

    # ---- split por FAZENDA ----
    rng = np.random.RandomState(42)
    farms = Counter(im["farm"] for im in images if im["source"] == "bahia")
    test_farm = args.test_farm
    mode = "farm-holdout"
    if not test_farm:
        if len(farms) >= 2:
            test_farm = min(farms, key=lambda k: farms[k])  # menor volume vira teste
        else:
            mode = "in-farm (otimista, 1 fazenda só)"

    train, val, test = [], [], []
    if mode == "farm-holdout":
        train_pool = []
        for im in images:
            if im["source"] == "bahia" and im["farm"] == test_farm:
                test.append(im["path"])
            else:
                train_pool.append(im["path"])
    else:
        # 1 fazenda: segura 20% da Bahia por classe (in-farm, otimista); ASDID todo no treino
        by_cls_bahia = defaultdict(list)
        train_pool = []
        for im in images:
            if im["source"] == "bahia":
                by_cls_bahia[im["class_id"]].append(im["path"])
            else:
                train_pool.append(im["path"])
        for cid, paths in by_cls_bahia.items():
            rng.shuffle(paths); k = max(1, int(0.2 * len(paths)))
            test.extend(paths[:k]); train_pool.extend(paths[k:])

    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * args.val_frac)
    val = train_pool[:n_val]; train = train_pool[n_val:]

    splits = {"_meta": {"strategy": mode, "test_farm": test_farm,
                        "note": "test = fazenda baiana não vista (proxy campo→campo real)."},
              "train": sorted(set(train)), "val": sorted(set(val)),
              "test": sorted(set(test)), "test_insource": []}
    json.dump(splits, open(out / "splits_region.json", "w"), indent=2, ensure_ascii=False)

    # ---- relatório ----
    print(f"\n{'='*66}\nDATASET COMBINADO: {out}\n{'='*66}")
    print(f"total={len(images)}  (asdid={sum(i['source']=='asdid' for i in images)}, "
          f"bahia={sum(i['source']=='bahia' for i in images)})")
    print(f"num_classes = {next_id}   <-- passe --num-classes {next_id} no treino")
    print(f"split: {mode}  test_farm={test_farm}")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"\n{'classe':22s} | asdid | bahia (por fazenda)")
    print("-" * 66)
    asdid_by_cls = Counter(id2name[i["class_id"]] for i in images if i["source"]=="asdid")
    for nm in sorted(dist, key=lambda n: dist[n]["id"]):
        b = dict(bahia_counts.get(nm, {}))
        tag = "  <- SEM Bahia (não testável cross-região)" if not b else ""
        print(f"{nm:22s} | {asdid_by_cls.get(nm,0):5d} | {b}{tag}")
    print(f"\nSalvo: {out}/manifest.json e {out}/splits_region.json")
    print("Treino: python scripts/train_field7.py --data-dir datasets/combined "
          f"--source-split --split-file splits_region.json --freeze-backbone --domain-aug "
          f"--num-classes {next_id}")


if __name__ == "__main__":
    main()
