# FastWAM DiT Flow MoT Design

This note documents the NavSim `FastWAMDiT` flow-supervision design and the mask rules used for training/eval.

## Current RGB-Flow Version

The current working tree uses official DPFlow RGB optical-flow images aligned to the front camera at `192x352`.

Data location:

```text
/mnt/zhaozc_workspace/data/fastwam_xty_dpflow_rgb_dataset_official_192x352
```

Each `.pt` cache item is expected to contain DPFlow outputs for the front camera pair:

- `flow_rgb`: `[3, 192, 352]`, RGB visualization image used as the training target.
- `flow_uv`: dense optical-flow field kept in the cache for inspection/debugging, but the model training path reads `flow_rgb`.

Training config points at this dataset through:

```yaml
generate_flow_gt: false
generate_flow_rgb: true
flow_rgb_cache_dir: /mnt/zhaozc_workspace/data/fastwam_xty_dpflow_rgb_dataset_official_192x352
video_size: [192, 352]
```

The current flow branch no longer uses the old `[B, 2, H, W]` BEV/UV `FlowDiT` path. Instead, `flow_rgb` is treated like a one-frame RGB video:

1. Dataset reads `sample["flow_rgb"]` and normalizes it to `[-1, 1]`.
2. `FastWAMDiT._encode_flow_rgb_latents` adds a temporal dimension and sends it through the same Wan VAE used by video.
3. The flow expert is a `WanVideoDiT`, configured with the same latent channel convention as video (`in_dim=48`, `out_dim=48`) and patching latent tokens rather than image pixels.
4. Flow loss is diffusion/flow-matching MSE in VAE latent space, not per-pixel token prediction.

This keeps token count comparable to video latents and avoids exploding into per-pixel RGB tokens.

## Current MoT Branches

The current model has three expert branches:

- `video`: original Wan/FastWAM video expert.
- `action`: original FastWAM action expert.
- `flow`: video-style `WanVideoDiT` expert over VAE-compressed DPFlow RGB images.

The intended interaction is still asymmetric:

```text
query \\ key | video                  action       flow
------------+-------------------------------------------
video       | original video mask     false        false
action      | first-frame video only  true         false
flow        | true                    true         true
```

So flow can use both video and action as context, but video/action cannot read flow tokens. This prevents flow supervision from leaking into action/video generation while still forcing the flow branch to align with the existing video/action token spaces.

## NavSim Eval Modes

NavSim evaluation now has explicit modes:

- `action_only`: calls `model.infer_action(...)`. This is the FastWAM-style action control path. It does not run joint video generation or flow generation, and it does not hide flow by zeroing masks.
- `joint`: keeps the original video+action joint path.
- `joint_flow`: calls `FastWAMDiT.infer_joint_flow(...)` and runs video, action, and flow experts together.
- `auto`: keeps `action_only` for `navtest`; otherwise defaults to `joint_flow`.

Evaluation dataset construction forcibly disables `generate_flow_gt` and `generate_flow_rgb`, because inference does not need training flow targets. This prevents action-only eval from being blocked by incomplete RGB-flow caches.

## Legacy Two-Channel Backup

The previous two-channel flow logic has been copied to:

```text
/mnt/zhaozc_workspace/project/FastWAM_xty_dit1
```

That backup restores the old key behavior:

- `flow_expert` is `FlowDiT`.
- Training reads `sample["flow_gt"]`.
- Flow target shape is `[B, 2, H, W]`.
- Flow tokens come from patchifying the 2-channel flow field directly.

Use `FastWAM_xty_dit1` if you need to rerun or compare the old logic. The current `FastWAM_xty_dit` directory is the RGB-flow VAE-latent three-branch version.

## Background Generation

Current RGB-flow generation is running under detached `setsid` processes. Logs are under:

```text
/mnt/zhaozc_workspace/project/FastWAM_xty_dit/outputs/dpflow_192x352_nohup/
```

The stable running shards are:

```text
start=0     max=21278
start=21278 max=21277
start=42555 max=21277
```

The final shard `start=63832 max=21277` should be started after resources are available if it is not already running. Four simultaneous workers exited during NavSim log loading, so three workers are the current safer setting.

## Legacy BEV/FlowDiT Notes

The sections below describe the earlier two-channel BEV/FlowDiT design and are kept only as historical reference.

## High-Level Structure

The model uses three DiT-style experts inside one MoT wrapper:

- `video`: the original Wan2.2 video expert loaded through `FastWAM.from_wan22_pretrained`.
- `action`: the existing FastWAM action expert.
- `flow`: a new `FlowDiT` expert for BEV motion-flow denoising.

The Wan2.2 video expert is not collapsed into a new shared DiT and its internal block structure is not replaced. Each expert keeps its own blocks, time modulation, RoPE frequencies, cross-attention, and MLP path. The MoT wrapper only mixes the experts at each layer's self-attention Q/K/V stage.

## Token Flow

During training, `FastWAMDiT.training_loss` builds each modality independently:

1. Video latents are noised with the video flow-matching scheduler and passed through `video_expert.pre_dit`.
2. Action tokens are noised with the action scheduler and passed through `action_expert.pre_dit`.
3. If `sample["flow_gt"]` exists, BEV flow is noised with the flow scheduler and passed through `flow_expert.pre_dit`.

`FlowDiT` tokenizes flow by patchifying `[B, 2, H, W]` BEV flow with a Conv2d patch embed. For the current config, `bev_size=[200, 200]` and `patch_size=4`, so flow produces `50 * 50 = 2500` flow tokens per sample.

The token dictionaries passed into `MoT.forward` are ordered by the registered expert order:

```text
[ video tokens ][ action tokens ][ flow/gt tokens ]
```

If flow GT is missing, only `video` and `action` are passed. `MoT.forward` now uses the active expert subset from `embeds_all`, so a model that has a registered flow expert will not crash when a batch/eval sample does not contain flow tokens.

## MoT Fusion Logic

For each transformer layer:

1. Each active expert builds its own Q/K/V from its own tokens and its own block parameters.
2. MoT concatenates all active Q/K/V tensors along the sequence dimension.
3. A single masked mixed-attention call is applied over the concatenated sequence.
4. The mixed attention output is split back into per-expert chunks.
5. Each expert applies its own output projection, residual gate, optional text cross-attention, and MLP.

This means flow is not predicted by a plain video/action head. Flow is predicted by the flow expert from noisy flow tokens, with video and action available as masked attention context.

## Attention Mask Rules

The flow-aware mask is built in `FastWAMDiT._build_mot_attention_mask_with_flow`.

Current rules:

- `video -> video`: uses the original FastWAM/Wan video mask via `video_expert.build_video_to_video_mask`, so video cannot see future frames.
- `video -> flow/gt`: disabled.
- `action -> action`: enabled.
- `action -> first-frame video`: enabled, matching original FastWAM behavior.
- `action -> flow/gt`: disabled.
- `flow/gt -> video`: enabled.
- `flow/gt -> action`: enabled.
- `flow/gt -> flow/gt`: enabled.

In matrix block form:

```text
query \\ key | video                  action       flow/gt
------------+----------------------------------------------
video       | original video mask     false        false
action      | first-frame video only  true         false
flow/gt     | true                    true         true
```

This satisfies the intended supervision direction: parallel GT branches can use video and action, while video/action cannot leak information from GT branches.

## Missing Flow Token Fix

The earlier distributed crash was:

```text
ValueError: Missing expert tokens for ['flow']
```

The cause was that `MoT` registered `flow` as an expert but still required all registered experts to be present in every forward call. Eval can produce a video/action-only call path when flow is absent, so this strict check was wrong.

The fix:

- `MoT.forward` computes `active_expert_order = [k for k in self.expert_order if k in embeds_all]`.
- Missing `freqs` and `t_mod` are checked only for active experts.
- The layer loop and output split loop iterate over `active_expert_order`.
- Eval batching keeps `flow_gt` and `flow_mask` when they exist.

As a result, both paths are valid:

- `video + action + flow`
- `video + action`

## Verification Commands

The following lightweight checks were run:

```bash
python -m py_compile \
  src/fastwam/models/wan22/mot.py \
  src/fastwam/models/wan22/fastwam_dit.py \
  src/fastwam/models/wan22/flow_dit.py \
  src/fastwam/trainer_tensorboard.py \
  src/fastwam/trainer.py
```

Additional regression checks covered:

- A `MoT` with registered `video/action/flow` experts can forward with only `video/action` tokens.
- The flow-aware attention mask matches the original video causal mask for the video block.
- `video` and `action` cannot attend to `flow/gt`.
- `flow/gt` can attend to `video`, `action`, and itself.
- `FlowDiT.pre_dit` and `post_dit` preserve expected token/image shapes.
- Eval batching preserves `flow_gt` and `flow_mask`.

## 32-GPU Training Notes

Use the 32-GPU task/script:

```bash
scripts/train_navsim_32gpu_dit.sh
```

The task config enables flow GT generation/cache:

```yaml
data:
  train:
    generate_flow_gt: true
    flow_cache_dir: ${oc.env:NAVSIM_FLOW_CACHE_DIR,...}
```

Before launching a long 32-GPU job, make sure the job is using the current workspace source. If a stack trace still shows `mot.py:457` raising `Missing expert tokens for ['flow']`, it is running stale code; in the current source line 457 is the unknown-token check, and the old exception string no longer exists.
