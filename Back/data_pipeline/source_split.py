"""
AtlasLeaf - Split por FONTE (avaliação honesta de generalização)
================================================================

Problema: cada doença está dominada por UMA fonte (asdid / kaggle / soynet).
Um split aleatório vaza a "assinatura da fonte" para treino e teste, inflando as
métricas. Este módulo constrói um split que:

  - Para classes com >=2 fontes: separa a MENOR fonte inteira para o teste
    (test_crosssource) -> mede generalização real para um domínio não visto.
  - Para classes com 1 fonte só: não há como medir cross-source; faz um holdout
    estratificado dentro da fonte (test_insource), claramente rotulado como
    otimista. Fica FORA do teste honesto.
  - O restante vira train/val estratificado.

Gera `splits_source.json` compatível com o loader (chaves train/val/test) e um
relatório detalhando o que é honesto vs. otimista.

Uso:
    python -m data_pipeline.source_split --data-dir datasets/unified
"""
import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np


def source_of(path: str) -> str:
    b = path.split("/")[-1]
    return re.split(r"_[0-9]", b)[0]


def build_split(
    data_dir: Path,
    min_source_for_test: int = 15,
    val_frac: float = 0.15,
    insource_test_frac: float = 0.15,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    manifest = json.load(open(data_dir / "manifest.json"))
    names = {v["id"]: k for k, v in manifest["class_distribution"].items()}

    # class_id -> source -> [rel_paths]
    by_cls_src = defaultdict(lambda: defaultdict(list))
    for img in manifest["images"]:
        cid = int(str(img["class_id"]))
        by_cls_src[cid][source_of(img["path"])].append(img["path"])

    train, val = [], []
    test_cross, test_insource = [], []
    report = []

    for cid in sorted(by_cls_src):
        srcs = by_cls_src[cid]
        # ordena fontes por tamanho (maior primeiro)
        ordered = sorted(srcs.items(), key=lambda kv: -len(kv[1]))
        total = sum(len(v) for _, v in ordered)

        # fonte(s) minoritária(s) elegíveis para teste cross-source
        held_out_sources = [s for s, v in ordered[1:] if len(v) >= min_source_for_test]

        kind = "single-source"
        if held_out_sources:
            kind = "cross-source"
            for s in held_out_sources:
                test_cross.extend(srcs[s])
            train_pool = []
            for s, v in ordered:
                if s not in held_out_sources:
                    train_pool.extend(v)
        else:
            # 1 fonte útil: holdout in-source (otimista)
            train_pool = [p for _, v in ordered for p in v]

        train_pool = list(train_pool)
        rng.shuffle(train_pool)

        if kind == "single-source":
            n_test = max(1, int(len(train_pool) * insource_test_frac))
            test_insource.extend(train_pool[:n_test])
            train_pool = train_pool[n_test:]

        n_val = max(1, int(len(train_pool) * val_frac))
        val.extend(train_pool[:n_val])
        train.extend(train_pool[n_val:])

        report.append({
            "class_id": cid,
            "class": names.get(cid, "?"),
            "sources": {s: len(v) for s, v in ordered},
            "eval_kind": kind,
            "held_out_sources": held_out_sources,
            "n_total": total,
        })

    splits = {
        "_meta": {
            "strategy": "source-aware",
            "note": "test = test_crosssource (honesto). test_insource é otimista e listado à parte.",
            "seed": seed,
        },
        "train": sorted(set(train)),
        "val": sorted(set(val)),
        # 'test' padrão = teste honesto cross-source (o que importa)
        "test": sorted(set(test_cross)),
        "test_insource": sorted(set(test_insource)),
        "report": report,
    }
    return splits, names


def print_report(splits):
    print(f"\n{'='*72}")
    print("SPLIT POR FONTE — relatório")
    print(f"{'='*72}")
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test_crosssource={len(splits['test'])}  test_insource={len(splits['test_insource'])}")
    print(f"\n{'cid':>3} | {'classe':22s} | {'avaliação':13s} | fontes")
    print("-" * 72)
    n_cross = n_single = 0
    for r in splits["report"]:
        if r["eval_kind"] == "cross-source":
            n_cross += 1
            tag = "HONESTO"
        else:
            n_single += 1
            tag = "otimista"
        print(f"{r['class_id']:>3} | {r['class']:22s} | {r['eval_kind']:13s} | {r['sources']}  ({tag})")
    print("-" * 72)
    print(f"Classes com teste HONESTO (cross-source): {n_cross}")
    print(f"Classes só com teste otimista (fonte única): {n_single}")
    print("\n⚠️  As métricas do 'test' honesto só cobrem as classes cross-source.")
    print("    As demais precisam de uma 2ª fonte de dados para serem avaliadas de verdade.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets/unified")
    ap.add_argument("--min-source-for-test", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits, _ = build_split(data_dir, min_source_for_test=args.min_source_for_test)
    out = Path(args.out) if args.out else data_dir / "splits_source.json"
    json.dump(splits, open(out, "w"), indent=2, ensure_ascii=False)
    print_report(splits)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
