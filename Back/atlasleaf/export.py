"""
AtlasLeaf - Export do modelo de campo (7 classes) para ONNX + metadata
======================================================================

Exporta o checkpoint field7 para ONNX e metadata.

Uso:
    python scripts/export_model.py [--ckpt artifacts/field7/model.pth]
"""
import argparse
import json
from pathlib import Path

import torch

from atlasleaf.artifacts import classifier_output_count, requires_output_mask, output_class_ids as checkpoint_output_class_ids
from atlasleaf.labels import FIELD7_CLASS_IDS, field7_classes
from atlasleaf.model import create_model
from atlasleaf.paths import FIELD7_ARTIFACT_DIR, from_back

BACK = Path(__file__).parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(FIELD7_ARTIFACT_DIR / "model.pth"))
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--num-classes", type=int, default=None,
                    help="Sobrescreve o nº de saídas; normalmente é inferido do checkpoint.")
    ap.add_argument("--onnx-out", default=str(FIELD7_ARTIFACT_DIR / "model.onnx"))
    ap.add_argument("--meta-out", default=str(FIELD7_ARTIFACT_DIR / "metadata.json"))
    ap.add_argument("--confidence-threshold", type=float, default=None,
                    help="Sobrescreve o limiar calibrado salvo no checkpoint.")
    args = ap.parse_args()

    ckpt_path = from_back(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    inferred_outputs = classifier_output_count(ck)
    num_classes = args.num_classes or inferred_outputs
    if num_classes != inferred_outputs:
        raise ValueError(f"--num-classes={num_classes}, mas o checkpoint tem {inferred_outputs} saídas")
    run = ck.get("run", {})
    output_class_ids = checkpoint_output_class_ids(ck, num_classes)
    masked_output = requires_output_mask(num_classes)

    print(f"🔄 Carregando {args.ckpt}...")
    model = create_model("efficientnet_v2_s", num_classes=num_classes,
                         pretrained=False, dropout=0.5)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"   checkpoint epoch={ck.get('epoch')} val_bal_acc={ck.get('val_bal_acc')}")

    print("🔄 Exportando ONNX...")
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    onnx_path = from_back(args.onnx_out)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, onnx_path,
        export_params=True, opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"✅ ONNX: {onnx_path}")

    classes = field7_classes()
    by_id = {int(c["id"]): c for c in classes}

    # nº de imagens do ASDID nas 7 classes suportadas (p/ o painel do app)
    total_images = None
    man_path = BACK / "datasets/unified/manifest.json"
    if man_path.exists():
        dist = json.load(open(man_path)).get("class_distribution", {})
        total_images = sum(v["count"] for v in dist.values() if int(v["id"]) in FIELD7_CLASS_IDS)

    calibration = ck.get("calibration", {})
    confidence_threshold = (args.confidence_threshold if args.confidence_threshold is not None
                            else calibration.get("threshold", 0.6))
    heldout_metrics = ck.get("test_metrics", {}).get("test", {})
    metadata = {
        "project": "AtlasLeaf",
        "version": "field7-1.0",
        "task": "soybean field disease classification (7 classes)",
        "model": "efficientnet_v2_s",
        "dataset": "asdid (7 classes de campo)",
        "total_images": total_images,
        "num_classes": num_classes,
        "supported_class_ids": FIELD7_CLASS_IDS,
        "output_class_ids": output_class_ids,
        "masked_output": masked_output,
        "classes": classes,
        "metrics": heldout_metrics,
        "preprocessing": {
            "resize": args.input_size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "leaf_crop": False,                    # recorte foi descartado (piorava)
        },
        "confidence_threshold": confidence_threshold,
        "calibration": calibration,
        "input_shape": [1, 3, args.input_size, args.input_size],
        "evaluation": {
            "protocol": "camera-holdout (treina Canon, testa Moto) — proxy campo→campo",
            **heldout_metrics,
            "note": "Deploy em lavoura diferente tende a ficar ABAIXO disso. Validar com fotos reais.",
        },
        "training": {
            "checkpoint": args.ckpt,
            "epoch": ck.get("epoch"),
            "val_balanced_acc": ck.get("val_bal_acc"),
            "config": ck.get("config", {}),
            "run": run,
        },
    }
    meta_path = from_back(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(metadata, open(meta_path, "w"), indent=2, ensure_ascii=False)
    print(f"✅ metadata: {meta_path}")
    print(f"   classes suportadas ({len(FIELD7_CLASS_IDS)}): "
          f"{[by_id[i]['friendly_name'] for i in FIELD7_CLASS_IDS]}")
    print(f"   limiar de confiança: {confidence_threshold:.3f}")
    if masked_output:
        print("⚠️ Checkpoint com saídas excedentes: retreine com --field7 antes de usar em produção.")


if __name__ == "__main__":
    main()
