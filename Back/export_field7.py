"""
AtlasLeaf - Export do modelo de campo (7 classes) para ONNX + metadata
======================================================================

Exporta o checkpoint do treino camera-split (atlasleaf_v31_sourcesplit_best.pth)
e gera uma metadata que marca APENAS as 7 classes de campo como suportadas.
O modelo tem 15 saídas (só 7 treinadas); o app/inferência mascaram o resto.

Uso:
    python export_field7.py [--ckpt atlasleaf_v31_sourcesplit_best.pth]
"""
import argparse
import json
from pathlib import Path

import torch

from data_pipeline.model_v31 import create_model

BACK = Path(__file__).parent

# 7 classes de campo (ASDID, com ambas as câmeras) — as únicas confiáveis p/ lavoura.
SUPPORTED_IDS = [0, 1, 2, 3, 7, 12, 13]

# Recall cross-camera (Canon->Moto) medido — vira "confiança esperada" por classe.
CROSS_CAMERA_RECALL = {
    0: 0.91, 1: 0.77, 2: 0.79, 3: 0.54, 7: 0.81, 12: 0.93, 13: 0.79,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="atlasleaf_v31_sourcesplit_best.pth")
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--num-classes", type=int, default=15)
    ap.add_argument("--onnx-out", default="atlasleaf_field7_diseases.onnx")
    ap.add_argument("--meta-out", default="atlasleaf_field7_metadata.json")
    ap.add_argument("--confidence-threshold", type=float, default=0.6)
    args = ap.parse_args()

    ckpt_path = BACK / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    print(f"🔄 Carregando {args.ckpt}...")
    model = create_model("efficientnet_v2_s", num_classes=args.num_classes,
                         pretrained=False, dropout=0.5)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"   checkpoint epoch={ck.get('epoch')} val_bal_acc={ck.get('val_bal_acc')}")

    print("🔄 Exportando ONNX...")
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    torch.onnx.export(
        model, dummy, BACK / args.onnx_out,
        export_params=True, opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"✅ ONNX: {args.onnx_out}")

    # Reaproveita nomes/severidade/científico da metadata de 15 classes
    base_meta = json.load(open(BACK / "atlasleaf_v31_metadata.json"))
    by_id = {int(c["id"]): c for c in base_meta["classes"]}

    supported_classes = []
    for cid in SUPPORTED_IDS:
        c = dict(by_id[cid])
        c["cross_camera_recall"] = CROSS_CAMERA_RECALL.get(cid)
        supported_classes.append(c)

    # nº de imagens do ASDID nas 7 classes suportadas (p/ o painel do app)
    total_images = None
    man_path = BACK / "datasets/unified/manifest.json"
    if man_path.exists():
        dist = json.load(open(man_path)).get("class_distribution", {})
        total_images = sum(v["count"] for v in dist.values() if int(v["id"]) in SUPPORTED_IDS)

    metadata = {
        "project": "AtlasLeaf",
        "version": "field7-1.0",
        "task": "soybean field disease classification (7 classes)",
        "model": "efficientnet_v2_s",
        "dataset": "asdid (7 classes de campo)",
        "total_images": total_images,
        "num_classes": args.num_classes,          # o modelo ainda tem 15 saídas
        "supported_class_ids": SUPPORTED_IDS,      # só estas são confiáveis
        "classes": base_meta["classes"],           # lista completa (indexável por id)
        "metrics": {"test_accuracy": 79.9, "test_balanced_acc": 79.3},
        "preprocessing": {
            "resize": args.input_size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "leaf_crop": False,                    # recorte foi descartado (piorava)
        },
        "confidence_threshold": args.confidence_threshold,
        "input_shape": [1, 3, args.input_size, args.input_size],
        "evaluation": {
            "protocol": "camera-holdout (treina Canon, testa Moto) — proxy campo→campo",
            "test_balanced_acc": 0.793,
            "test_acc": 0.799,
            "n_test": 2655,
            "note": "Deploy em lavoura diferente tende a ficar ABAIXO disso. Validar com fotos reais.",
        },
        "training": {
            "checkpoint": args.ckpt,
            "epoch": ck.get("epoch"),
            "val_balanced_acc": ck.get("val_bal_acc"),
            "recipe": "unified (sem recorte) + freeze-backbone + domain-aug + sampler sqrt",
        },
    }
    json.dump(metadata, open(BACK / args.meta_out, "w"), indent=2, ensure_ascii=False)
    print(f"✅ metadata: {args.meta_out}")
    print(f"   classes suportadas ({len(SUPPORTED_IDS)}): "
          f"{[by_id[i]['friendly_name'] for i in SUPPORTED_IDS]}")
    print(f"   limiar de confiança: {args.confidence_threshold}")


if __name__ == "__main__":
    main()
