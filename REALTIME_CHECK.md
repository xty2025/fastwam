# FastWAM NAVSIM Realtime Check

## Current status

Current run:

```text
/mnt/zhaozc_workspace/project/FastWAM_xty_dit/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28
```

Checkpoint status right now:

- `checkpoints/weights/`: `0` saved weight checkpoints
- `checkpoints/state/`: `0` saved trainer-state checkpoints

Why:

- current config uses `save_every=10000`
- this run only reached about `step 880`
- so no checkpoint has been written yet

Resource status when this note was generated:

- RAM: about `46 GiB available`
- GPU memory: not checked here for a live process
- disk `/mnt/zhaozc_workspace`: `100% used`
- disk `/mnt/data`: `100% used`

Important:

- RAM is currently fine
- disk is not fine
- even if training keeps going, checkpoint save and eval output are both at high risk of failing because the shared storage is full

## What metric to look at

Training log eval:

- `val_loss`
- `action_l1`
- `action_l2`
- `psnr`
- `ssim`

Offline NAVSIM eval:

- `action_l1`
- `action_l2`
- `traj_ade`
- `traj_fde`
- `heading_mae`

Offline PDM score:

- final CSV contains a `score` column
- this is the number most likely corresponding to values like:

```text
Result 74.72 / 3K
Result 89.49 / 22K
Result 90.15 / 33K
Result 90.11 / 44K
Result 90.02 / 44.4K
```

Here `3K / 22K / 33K / 44K / 44.4K` are checkpoint steps.

## Manual check for one checkpoint

Use this when a checkpoint file already exists under:

```text
.../checkpoints/weights/step_XXXXXX.pt
```

Run from anywhere:

```bash
bash -lc '
  set -e
  export PROJECT_ROOT=/mnt/zhaozc_workspace/project/FastWAM_xty_dit
  export RUN_DIR=$PROJECT_ROOT/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28
  export CKPT=$RUN_DIR/checkpoints/weights/step_003000.pt
  export TASK=navsim_uncond_front_384x672_xty_dit_16gpu
  export SPLIT=navsim
  export EXPERIMENT_NAME=$(basename "${CKPT%.pt}")
  export NPROC_PER_NODE=2
  export MASTER_PORT=27890

  cd "$PROJECT_ROOT"
  bash experiments/navsim/run_eval_then_pdm_current.sh
'
```

Notes:

- `SPLIT=navsim`: local validation-style split
- `SPLIT=navtest`: test split
- `NPROC_PER_NODE=2` is usually enough for eval

Output will be written under:

```text
$PROJECT_ROOT/evaluate_results/navsim/$TASK/$EXPERIMENT_NAME/<timestamp>/
```

Files to inspect:

- `summary.json`
- `per_sample_results.jsonl`
- `pred_actions/*.npy`
- `pdm_score/*.csv`

## Read the final score quickly

After eval finishes:

```bash
bash -lc '
  export OUT_DIR=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/evaluate_results/navsim/navsim_uncond_front_384x672_xty_dit_16gpu/step_003000
  export LATEST=$(find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)
  export CSV=$(find "$LATEST/pdm_score" -name "*.csv" | sort | tail -n 1)
  python - <<PY
import pandas as pd
csv = r"$CSV"
df = pd.read_csv(csv)
print(df.tail(1).to_string(index=False))
PY
'
```

The last row is the average row. The main field to watch is `score`.

## Realtime auto-check for every new checkpoint

This loop waits for new weight checkpoints and evaluates each one once.

```bash
bash -lc '
  set -e
  export PROJECT_ROOT=/mnt/zhaozc_workspace/project/FastWAM_xty_dit
  export RUN_DIR=$PROJECT_ROOT/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28
  export TASK=navsim_uncond_front_384x672_xty_dit_16gpu
  export SPLIT=navsim
  export NPROC_PER_NODE=2
  export DONE_DIR=$RUN_DIR/.eval_done
  mkdir -p "$DONE_DIR"
  cd "$PROJECT_ROOT"

  while true; do
    found=0
    for ckpt in "$RUN_DIR"/checkpoints/weights/step_*.pt; do
      [ -e "$ckpt" ] || continue
      found=1
      step=$(basename "${ckpt%.pt}")
      done_flag="$DONE_DIR/$step.done"
      [ -f "$done_flag" ] && continue

      MASTER_PORT=$((27890 + RANDOM % 1000)) \
      CKPT="$ckpt" \
      TASK="$TASK" \
      SPLIT="$SPLIT" \
      EXPERIMENT_NAME="$step" \
      NPROC_PER_NODE="$NPROC_PER_NODE" \
      bash experiments/navsim/run_eval_then_pdm_current.sh

      touch "$done_flag"
    done

    if [ "$found" -eq 0 ]; then
      echo "[watch] no checkpoint yet"
    fi
    sleep 300
  done
'
```

This is the simplest way to do "save then check".

## Check how many checkpoints exist

```bash
bash -lc '
  export RUN_DIR=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28
  echo "weight checkpoints: $(find "$RUN_DIR/checkpoints/weights" -maxdepth 1 -name "step_*.pt" | wc -l)"
  echo "state checkpoints:  $(find "$RUN_DIR/checkpoints/state" -maxdepth 1 -type d -name "step_*" | wc -l)"
'
```

## Check if memory/storage is enough

```bash
bash -lc '
  free -h
  df -h /mnt/zhaozc_workspace /mnt/data /
'
```

Interpretation:

- if `available` RAM is still large, host memory is probably fine
- if `/mnt/zhaozc_workspace` or `/mnt/data` is `100%`, checkpoint save and eval output may fail

## Practical recommendation

For this run, realtime check is currently blocked by two facts:

1. no checkpoint has been saved yet
2. shared storage is already full

To make realtime check actually work:

1. free disk space first
2. let training reach the first save point, or reduce `save_every`
3. then use the manual command or the watch loop above

