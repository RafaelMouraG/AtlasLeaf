#!/usr/bin/env bash
# Teima cross-região (ASDID -> Mendeley V2 holdout): compara fine-tune completo
# da EfficientNet-V2-S (08) vs ConvNeXt-Tiny (09) em fotos de outro país/região.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$BACK_DIR/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
DATA_DIR="$BACK_DIR/datasets/unified_resized"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BACK_DIR/artifacts/runs/cross_region_$STAMP"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/cross_region.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "🌍 AtlasLeaf — Teima Cross-Região (ASDID -> Mendeley V2) ($STAMP)"
echo "   Saída: $RUN_DIR"
echo "========================================================================"
cd "$BACK_DIR"

run_exp() {
  local name="$1"
  shift
  echo ""
  echo "------------------------------------------------------------------------"
  echo "🚀 [$(date '+%H:%M:%S')] Iniciando: $name"
  echo "------------------------------------------------------------------------"
  local t0
  t0="$(date +%s)"
  if "$PYTHON" scripts/train_field7.py \
      --data-dir "$DATA_DIR" \
      --source-split \
      --split-file splits_mendeley_holdout.json \
      --field7 \
      --domain-aug \
      --amp \
      --num-workers 6 \
      --epochs 40 \
      --batch-size 16 \
      --accumulation-steps 3 \
      --lr 1e-4 \
      --target-precision 0.99 \
      --ckpt-out "$RUN_DIR/${name}.pth" \
      "$@"; then
    local dt=$(( $(date +%s) - t0 ))
    echo "✅ [$(date '+%H:%M:%S')] Concluído: $name (${dt}s)"
  else
    echo "⚠️ [$(date '+%H:%M:%S')] Falhou: $name"
  fi
}

run_exp "08_effv2s_ft_mendeley_holdout" --model efficientnet_v2_s
run_exp "09_convnext_ft_mendeley_holdout" --model convnext_tiny

echo ""
echo "========================================================================"
echo "📊 RESUMO CROSS-REGIÃO (holdout Mendeley V2 — 295 imgs nunca vistas)"
echo "========================================================================"
"$PYTHON" - "$RUN_DIR" <<'PYEOF'
import sys, pathlib, torch, pandas as pd

run_dir = pathlib.Path(sys.argv[1])
rows = []
for pth in sorted(run_dir.glob("*.pth")):
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    tm = ck.get("test_metrics", {}).get("test", {})
    pc = tm.get("per_class", {})
    rows.append({
        "exp": pth.stem,
        "model": cfg.get("model_name", "?"),
        "epoch": (ck.get("epoch") + 1) if isinstance(ck.get("epoch"), int) else None,
        "val_bal": round(ck.get("val_bal_acc", 0.0), 2),
        "mendeley_bal": round(tm.get("test_balanced_acc", 0.0), 2) if tm else None,
        "mendeley_acc": round(tm.get("test_accuracy", 0.0), 2) if tm else None,
        "healthy_rec": round(pc.get("healthy", {}).get("recall", 0) * 100, 1),
        "rust_rec": round(pc.get("asian_rust", {}).get("recall", 0) * 100, 1),
        "cerc_rec": round(pc.get("cercospora_blight", {}).get("recall", 0) * 100, 1),
    })

if rows:
    df = pd.DataFrame(rows).sort_values("mendeley_bal", ascending=False)
    df.to_csv(run_dir / "summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n✅ Tabela salva em: {run_dir / 'summary.csv'}")
PYEOF
