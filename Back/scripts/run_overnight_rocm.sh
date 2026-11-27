#!/usr/bin/env bash
# Bateria de experimentos AtlasLeaf (field7) otimizada para GPU AMD ROCm (RX 7800 XT).
# Salva checkpoints e logs em Back/artifacts/runs/overnight_<timestamp>/ sem sobrescrever produção.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$BACK_DIR/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
DATA_DIR="$BACK_DIR/datasets/unified_resized"

# Usa apenas a GPU dedicada AMD (GPU[0]: RX 7800 XT), ignorando a iGPU do Ryzen
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON" ]]; then
  echo "❌ Ambiente virtual não encontrado em $PYTHON"
  exit 1
fi
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "❌ Dataset não encontrado em $DATA_DIR"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BACK_DIR/artifacts/runs/overnight_$STAMP"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/overnight.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "🌿 AtlasLeaf — Bateria Overnight ROCm ($STAMP)"
echo "   Diretório de saída : $RUN_DIR"
echo "   HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "========================================================================"
cd "$BACK_DIR"

run_exp() {
  local name="$1"
  shift
  echo ""
  echo "------------------------------------------------------------------------"
  echo "🚀 [$(date '+%H:%M:%S')] Iniciando: $name"
  echo "   Args: $*"
  echo "------------------------------------------------------------------------"
  local t0
  t0="$(date +%s)"
  if "$PYTHON" scripts/train_field7.py \
      --data-dir "$DATA_DIR" \
      --source-split \
      --field7 \
      --domain-aug \
      --amp \
      --num-workers 6 \
      --ckpt-out "$RUN_DIR/${name}.pth" \
      "$@"; then
    local dt=$(( $(date +%s) - t0 ))
    echo "✅ [$(date '+%H:%M:%S')] Concluído: $name (${dt}s)"
  else
    echo "⚠️ [$(date '+%H:%M:%S')] Falhou: $name (continuando próximo experimento)"
  fi
}

# 1. Baseline oficial (Canon -> Moto), backbone congelado
run_exp "01_effv2s_frozen_camera" \
  --model efficientnet_v2_s --split-file splits_camera.json \
  --freeze-backbone --epochs 40 --batch-size 32 --accumulation-steps 2 --lr 5e-4

# 2. Mesmo baseline + Mendeley V2 no treino (mesmo holdout Moto)
run_exp "02_effv2s_frozen_camera_mendeley" \
  --model efficientnet_v2_s --split-file splits_camera_mendeley_train.json \
  --freeze-backbone --epochs 40 --batch-size 32 --accumulation-steps 2 --lr 5e-4

# 3. ConvNeXt-Tiny congelado (Canon -> Moto)
run_exp "03_convnext_frozen_camera" \
  --model convnext_tiny --split-file splits_camera.json \
  --freeze-backbone --epochs 40 --batch-size 32 --accumulation-steps 2 --lr 5e-4

# 4. ConvNeXt-Tiny congelado + Mendeley V2 (Canon+Mendeley -> Moto)
run_exp "04_convnext_frozen_camera_mendeley" \
  --model convnext_tiny --split-file splits_camera_mendeley_train.json \
  --freeze-backbone --epochs 40 --batch-size 32 --accumulation-steps 2 --lr 5e-4

# 5. ConvNeXt-Tiny fine-tune completo (LR baixo) + Mendeley V2
run_exp "05_convnext_ft_camera_mendeley" \
  --model convnext_tiny --split-file splits_camera_mendeley_train.json \
  --epochs 40 --batch-size 16 --accumulation-steps 3 --lr 1e-4

# 6. EfficientNet-V2-S fine-tune completo (LR baixo) + Mendeley V2
run_exp "06_effv2s_ft_camera_mendeley" \
  --model efficientnet_v2_s --split-file splits_camera_mendeley_train.json \
  --epochs 40 --batch-size 16 --accumulation-steps 3 --lr 1e-4

# 7. Teste cross-região duro: treina ASDID inteiro, testa só no Mendeley V2
run_exp "07_convnext_frozen_mendeley_holdout" \
  --model convnext_tiny --split-file splits_mendeley_holdout.json \
  --freeze-backbone --epochs 40 --batch-size 32 --accumulation-steps 2 --lr 5e-4

echo ""
echo "========================================================================"
echo "📊 RESUMO FINAL DA BATERIA ($RUN_DIR)"
echo "========================================================================"
"$PYTHON" - "$RUN_DIR" <<'PYEOF'
import sys, pathlib, torch, pandas as pd

run_dir = pathlib.Path(sys.argv[1])
rows = []
for pth in sorted(run_dir.glob("*.pth")):
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    run = ck.get("run", {})
    test_m = ck.get("test_metrics", {}).get("test", {})
    sel = test_m.get("selective", {})
    cerc = test_m.get("per_class", {}).get("cercospora_blight", {}).get("recall")
    rows.append({
        "exp": pth.stem,
        "model": cfg.get("model_name", "?"),
        "split": run.get("split_file", "?"),
        "epoch": (ck.get("epoch") + 1) if isinstance(ck.get("epoch"), int) else None,
        "val_bal": round(ck.get("val_bal_acc", 0.0), 2),
        "test_bal": round(test_m.get("test_balanced_acc", 0.0), 2) if test_m else None,
        "test_acc": round(test_m.get("test_accuracy", 0.0), 2) if test_m else None,
        "cerc_rec": round(cerc * 100, 1) if cerc is not None else None,
        "thr": round(sel.get("threshold", 0.0), 3) if sel else None,
        "cov": round(sel.get("coverage", 0.0) * 100, 1) if sel else None,
        "sel_acc": round(sel.get("accuracy", 0.0), 1) if sel and sel.get("accuracy") is not None else None,
    })

if rows:
    df = pd.DataFrame(rows).sort_values("test_bal", ascending=False)
    df.to_csv(run_dir / "summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n✅ Tabela salva em: {run_dir / 'summary.csv'}")
else:
    print("Nenhum checkpoint salvo.")
PYEOF
