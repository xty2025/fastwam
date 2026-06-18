# FastWAM xty DiT: Results, Navsim Evaluation, TensorBoard

## Current Saved Results

Project directory:

```bash
cd /mnt/zhaozc_workspace/project/FastWAM_xty_dit
```

Current run:

```text
runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28
```

Saved checkpoint:

```text
runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/checkpoints/weights/step_010000.pt
```

DeepSpeed resume state:

```text
runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/checkpoints/state/step_010000
```

Saved qualitative eval videos:

```text
eval mp4 count: 48
step_003000: 16 videos
step_006000: 16 videos
step_009000: 16 videos
```

Video directory:

```text
runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/eval
```

As of this check, there is no completed Navsim `pred_actions` or `pdm_score` output under this repo yet. The commands below generate those from the saved `step_010000.pt`.

## Environment

Use the project root:

```bash
cd /mnt/zhaozc_workspace/project/FastWAM_xty_dit
```

Set Python import paths:

```bash
export PYTHONPATH=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/src:/mnt/zhaozc_workspace/project/FastWAM_xty_dit/navsim:/mnt/zhaozc_workspace/project/FastWAM_xty_dit:${PYTHONPATH:-}
```

Set Navsim data paths:

```bash
export NAVSIM_ROOT=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/data/navsim
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=${NAVSIM_ROOT}/maps
export NAVSIM_EXP_ROOT=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/evaluate_results
export NAVSIM_DEVKIT_ROOT=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/navsim
export OPENSCENE_DATA_ROOT=${NAVSIM_ROOT}
```

The metric cache currently exists in the older FastWAM directory:

```text
/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtest
/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtrain
```

The bundled script below automatically falls back to `/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_${SPLIT}` if the cache is not present in `FastWAM_xty_dit/data`.

## Recommended Full Navsim Metric Test

This runs model inference, writes `pred_actions`, then runs Navsim PDM scoring.

Default target:

```text
task: navsim_uncond_front_384x672_xty_dit_16gpu
ckpt: step_010000.pt
split: navtest
```

Run:

```bash
cd /mnt/zhaozc_workspace/project/FastWAM_xty_dit

TASK=navsim_uncond_front_384x672_xty_dit_16gpu \
CKPT=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/checkpoints/weights/step_010000.pt \
EXPERIMENT_NAME=step_010000_navtest \
SPLIT=navtest \
NPROC_PER_NODE=1 \
MASTER_PORT=27890 \
SAVE_VIDEOS=false \
bash experiments/navsim/run_eval_then_pdm_current.sh
```

If you want to use 2 GPUs, change:

```bash
NPROC_PER_NODE=2
```

If port `27890` is occupied, use another port:

```bash
MASTER_PORT=27901
```

## Quick Smoke Test

Before running the full `navtest`, test a small sample count:

```bash
cd /mnt/zhaozc_workspace/project/FastWAM_xty_dit

TASK=navsim_uncond_front_384x672_xty_dit_16gpu \
CKPT=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/checkpoints/weights/step_010000.pt \
EXPERIMENT_NAME=step_010000_navtest_smoke \
SPLIT=navtest \
NPROC_PER_NODE=1 \
MASTER_PORT=27890 \
SAVE_VIDEOS=true \
MAX_SAMPLES=16 \
bash experiments/navsim/run_eval_then_pdm_current.sh
```

The smoke test writes a small output under:

```text
evaluate_results/navsim/navsim_uncond_front_384x672_xty_dit_16gpu/step_010000_navtest_smoke/<timestamp>
```

Check the generated files:

```bash
find evaluate_results/navsim/navsim_uncond_front_384x672_xty_dit_16gpu/step_010000_navtest_smoke -maxdepth 4 -type f | sort | tail -50
```

## Output Files

After a successful run, the output directory is printed at the end:

```text
[done] eval output: <OUTPUT_DIR>
[done] pdm csv dir: <OUTPUT_DIR>/pdm_score
```

Important files:

```text
<OUTPUT_DIR>/summary.json
<OUTPUT_DIR>/per_sample_results.jsonl
<OUTPUT_DIR>/pred_actions/*.npy
<OUTPUT_DIR>/pdm_score
```

`summary.json` contains FastWAM-side prediction metrics such as:

```text
action_l1_mean
action_l2_mean
traj_ade_mean
traj_fde_mean
heading_mae_mean
video_psnr_mean
video_ssim_mean
```

`pdm_score` contains Navsim PDM evaluation outputs. Use:

```bash
find <OUTPUT_DIR>/pdm_score -type f | sort
```

to locate the produced CSV/JSON files.

## Manual Two-Step Evaluation

If you want to separate FastWAM inference and Navsim PDM scoring, run the two steps manually.

### Step 1: Generate Predicted Actions

```bash
cd /mnt/zhaozc_workspace/project/FastWAM_xty_dit

OUTPUT_DIR=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/evaluate_results/navsim/navsim_uncond_front_384x672_xty_dit_16gpu/step_010000_navtest/manual_$(date +%Y%m%d_%H%M%S)

torchrun \
  --nproc_per_node 1 \
  --nnodes 1 \
  --master-port 27890 \
  experiments/navsim/eval_navsim.py \
  task=navsim_uncond_front_384x672_xty_dit_16gpu \
  ckpt=/mnt/zhaozc_workspace/project/FastWAM_xty_dit/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/checkpoints/weights/step_010000.pt \
  experiment_name=step_010000_navtest \
  EVALUATION.split=navtest \
  EVALUATION.output_dir=${OUTPUT_DIR} \
  EVALUATION.save_videos=false
```

### Step 2: Run PDM Score

```bash
python /mnt/zhaozc_workspace/project/FastWAM_xty_dit/navsim/navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  metric_cache_path=/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtest \
  agent=npy_trajectory_agent \
  agent.pred_actions_path=${OUTPUT_DIR}/pred_actions \
  experiment_name=step_010000_navtest_pdm \
  output_dir=${OUTPUT_DIR}/pdm_score
```

## TensorBoard

TensorBoard event file:

```text
runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/train/events.out.tfevents.*
```

If TensorBoard is not installed in the active Python environment:

```bash
pip install tensorboard
```

Start TensorBoard:

```bash
cd /mnt/zhaozc_workspace/project/FastWAM_xty_dit

tensorboard \
  --logdir runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/train \
  --host 0.0.0.0 \
  --port 6006
```

If port `6006` is occupied:

```bash
tensorboard \
  --logdir runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/train \
  --host 0.0.0.0 \
  --port 6007
```

On a remote server, forward the port from your local machine:

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<server>
```

Then open:

```text
http://127.0.0.1:6006
```

## Progress From Logs

Current training progress can also be checked with:

```bash
tail -f /mnt/zhaozc_workspace/project/FastWAM_xty_dit/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/train.log
```

Latest checkpoint list:

```bash
find /mnt/zhaozc_workspace/project/FastWAM_xty_dit/runs/navsim_uncond_front_384x672_xty_dit_16gpu/2026-06-09_16-17-28/checkpoints/weights -type f -name '*.pt' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```
