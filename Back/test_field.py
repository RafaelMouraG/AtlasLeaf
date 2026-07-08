"""
AtlasLeaf - Teste com fotos REAIS da sua lavoura
================================================

O número que decide se dá pra colocar no ar. Usa o ONNX de campo exatamente como
o deploy (resize 384, sem recorte, filtro das 7 classes + limiar de confiança).

Organize suas fotos em subpastas com o NOME da classe:

    minhas_fotos/
        healthy/            *.jpg
        asian_rust/         *.jpg
        cercospora_blight/  *.jpg
        ...

Uso:
    python test_field.py --dir minhas_fotos
    python test_field.py --dir minhas_fotos --onnx atlasleaf_field7_diseases.onnx
"""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

BACK = Path(__file__).parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess(img: Image.Image, size, mean, std):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32) / 255.0
    a = a.transpose(2, 0, 1)
    a = (a - np.array(mean).reshape(3, 1, 1)) / np.array(std).reshape(3, 1, 1)
    return a[None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Pasta com subpastas por classe")
    ap.add_argument("--onnx", default="atlasleaf_field7_diseases.onnx")
    ap.add_argument("--meta", default="atlasleaf_field7_metadata.json")
    args = ap.parse_args()

    meta = json.load(open(BACK / args.meta))
    prep = meta["preprocessing"]
    size = prep["resize"]
    supported = meta["supported_class_ids"]
    by_id = {int(c["id"]): c for c in meta["classes"]}
    name2id = {c["name"]: int(c["id"]) for c in meta["classes"]}
    threshold = float(meta.get("confidence_threshold", 0.6))

    sess = ort.InferenceSession(str(BACK / args.onnx))
    inp = sess.get_inputs()[0].name

    root = Path(args.dir)
    y_true, y_pred, confs, skipped = [], [], [], Counter()
    n_uncertain = 0

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = sub.name
        if cls not in name2id or name2id[cls] not in supported:
            skipped[cls] += sum(1 for f in sub.rglob("*") if f.suffix.lower() in IMG_EXT)
            continue
        cid = name2id[cls]
        for f in sub.rglob("*"):
            if f.suffix.lower() not in IMG_EXT:
                continue
            try:
                x = preprocess(Image.open(f), size, prep["mean"], prep["std"])
            except Exception as e:
                print(f"  [falha] {f.name}: {e}")
                continue
            logits = sess.run(None, {inp: x})[0][0]
            # softmax só nas classes suportadas
            sl = np.array([logits[i] for i in supported], dtype=np.float64)
            sl = np.exp(sl - sl.max())
            probs = sl / sl.sum()
            k = int(np.argmax(probs))
            pred_id = supported[k]
            conf = float(probs[k])
            y_true.append(cid)
            y_pred.append(pred_id)
            confs.append(conf)
            if conf < threshold:
                n_uncertain += 1

    if not y_true:
        print("Nenhuma imagem encontrada nas subpastas das 7 classes suportadas.")
        if skipped:
            print("Pastas ignoradas (classe não suportada):", dict(skipped))
        return

    y_true, y_pred, confs = np.array(y_true), np.array(y_pred), np.array(confs)
    present = sorted(set(y_true) | set(y_pred))
    names = [by_id[c]["friendly_name"] for c in present]

    print(f"\n{'='*60}\nTESTE DE CAMPO — {len(y_true)} imagens\n{'='*60}")
    print(f"Acurácia:          {(y_true==y_pred).mean()*100:.1f}%")
    print(f"Acurácia balanceada: {balanced_accuracy_score(y_true, y_pred)*100:.1f}%")
    print(f"Confiança média:   {confs.mean()*100:.1f}%")
    print(f"Sinalizadas incertas (<{threshold*100:.0f}%): {n_uncertain}/{len(y_true)} "
          f"({100*n_uncertain/len(y_true):.0f}%)")

    # acurácia SÓ nas confiantes (o que o produto de fato entregaria)
    conf_mask = confs >= threshold
    if conf_mask.sum():
        acc_conf = (y_true[conf_mask] == y_pred[conf_mask]).mean() * 100
        print(f"Acurácia nas CONFIANTES: {acc_conf:.1f}% (n={conf_mask.sum()}) "
              f"— o resto seria enviado p/ revisão humana")

    print("\n" + classification_report(y_true, y_pred, labels=present,
                                        target_names=names, zero_division=0))
    print("Matriz de confusão (linha=verdadeiro, coluna=predito):")
    print("labels:", [by_id[c]["name"] for c in present])
    print(confusion_matrix(y_true, y_pred, labels=present))
    if skipped:
        print("\n⚠️ Pastas ignoradas (classe não suportada pelo modelo de 7):", dict(skipped))


if __name__ == "__main__":
    main()
