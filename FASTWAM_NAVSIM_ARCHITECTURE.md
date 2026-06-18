# FastWAM-XTY: From Robot World Action Model to NAVSIM Autonomous Driving

本文档记录当前仓库中 FastWAM 如何从机器人任务迁移到 NAVSIM 自驾轨迹预测任务。重点包括整体框架、数据与动作语义变化、核心模块构建、训练/推理/评测链路、目录功能和当前代码改动范围。

## 1. 总体结论

原始 FastWAM 主要面向机器人操作任务，例如 LIBERO 和 RoboTwin。它的抽象不是“机器人专用模型”，而是一个 World Action Model：

```text
当前视觉观测 + 文本任务条件 + 本体状态条件
        -> 未来视频想象 + 未来动作 chunk
```

在机器人任务中，动作通常是机械臂关节、末端执行器或夹爪控制量。在当前 FastWAM-XTY 仓库中，这个动作空间被替换为 NAVSIM 的未来 ego trajectory：

```text
action[t] = [x, y, heading]
```

因此，自驾迁移的核心不是重写整套模型，而是把 FastWAM 的“未来动作 chunk”语义从机器人控制量换成车辆未来轨迹，同时把输入观测、状态、prompt、评测方式切到 NAVSIM。

当前 NAVSIM 版本主要实现的是：

- 输入：当前前视相机图像、ego 状态、驾驶指令、文本 prompt。
- 输出：未来 8 步 ego trajectory，每步 3 维 `[x, y, heading]`。
- 训练：仍保留 video branch + action branch 的联合 flow matching 训练。
- 推理：NAVSIM test 默认更常用 `infer_action()`，只输出 trajectory，不必生成未来视频。
- 评测：输出 `.npy` 轨迹，计算 ADE、FDE、heading MAE，也可在 joint 模式下计算视频 PSNR/SSIM。

## 2. 原始 FastWAM 框架

FastWAM 的核心由四部分组成：

```text
Wan VAE
  RGB video <-> latent video

Wan Video DiT
  对视频 latent 做 diffusion / flow matching 建模

ActionDiT
  对连续动作序列做 diffusion / flow matching 建模

MoT, Mixture of Transformers
  把 video expert 和 action expert 按层混合，让动作 token 可以读取视觉 token
```

在代码中对应：

- `src/fastwam/models/wan22/wan_video_vae.py`
- `src/fastwam/models/wan22/wan_video_dit.py`
- `src/fastwam/models/wan22/action_dit.py`
- `src/fastwam/models/wan22/mot.py`
- `src/fastwam/models/wan22/fastwam.py`

原始机器人任务中的典型数据语义：

```text
video:      多相机机器人观测
action:     机器人动作 chunk
proprio:    机器人本体状态
prompt:     任务语言指令
```

FastWAM 的训练目标是对未来视频 latent 和动作序列同时加噪，然后训练模型预测 flow matching target。

## 3. 迁移到 NAVSIM 的核心思想

NAVSIM 自驾任务可以被重新表述为同一种 World Action Model：

```text
当前车载相机图像 + 当前 ego 状态 + 导航/驾驶命令
        -> 未来驾驶视频 + 未来 ego trajectory
```

迁移时主要替换了四个层面的语义。

### 3.1 观测从机器人相机变为车载相机

NAVSIM 数据由 `NavSimVideoDataset` 读取：

```text
src/fastwam/datasets/navsim/navsim_dataset.py
```

它使用 NAVSIM 的：

- `SceneLoader`
- `SceneFilter`
- `SensorConfig`

构造样本。当前支持两种 camera layout：

```text
front
  只用 cam_f0 前视相机

stitched_front
  拼接 cam_l0 + cam_f0 + cam_r0，形成更宽的前向视野
```

配置在：

```text
configs/data/navsim.yaml
```

当前主线常用：

```yaml
num_frames: 9
future_action_horizon: 8
video_frame_mode: current_plus_future
camera_layout: front
video_size: [768, 1344]   # 或 384x672 / 384x512 / 192x352
```

因此一个训练 sample 的 video 形状是：

```text
video: [3, 9, H, W]
```

进入 DataLoader batch 后：

```text
video: [B, 3, 9, H, W]
```

### 3.2 动作从机器人控制量变为车辆未来轨迹

NAVSIM 中的动作来自：

```python
scene.get_future_trajectory(num_trajectory_frames=future_action_horizon).poses
```

在 `NavSimVideoDataset._extract_future_trajectory()` 中实现。

动作含义：

```text
action: [T, 3]
T = future_action_horizon = 8
3 = [x, y, heading]
```

对应配置：

```yaml
action_dim: 3
future_action_horizon: 8
```

模型配置中也同步把 ActionDiT 改成 3 维输入输出：

```yaml
configs/model/fastwam_navsim.yaml

video_dit_config:
  action_dim: 3

action_dit_config:
  action_dim: 3
```

NAVSIM 数据集对轨迹做了固定范围归一化：

```python
x       -> [-1, 1]
y       -> [-1, 1]
heading -> [-1, 1]
```

代码在：

```text
NavSimVideoDataset.norm_odo()
NavSimVideoDataset.denorm_odo()
NavSimVideoDataset.denormalize_action()
```

注意：`configs/data/navsim.yaml` 里仍保留 `pretrained_norm_stats` / `stats_cache_path` 字段，但 `NavSimVideoDataset` 当前会忽略它们，实际使用固定 odo bounds 归一化。

### 3.3 Proprio 从机器人状态变为 ego 状态

NAVSIM 的 proprio/state 在 `NavSimVideoDataset._extract_ego_features()` 中构造：

```text
velocity:         2 维
acceleration:     2 维
driving_command:  4 维 one-hot
----------------------
state_dim:        8 维
```

dataset 返回：

```text
state:   [T, 8]
proprio: [T, 8]
```

模型构造时：

```yaml
proprio_dim: ${data.train.state_dim}
```

在 `FastWAM.__init__()` 中，如果 `proprio_dim` 不为空，会创建：

```python
self.proprio_encoder = nn.Linear(proprio_dim, text_dim)
```

训练和推理时取当前时刻的 proprio：

```python
proprio = proprio[:, 0, :]   # [B, 8]
```

再映射成一个 context token，拼到文本 embedding 后面：

```text
context:      [B, L, text_dim]
proprio_tok:  [B, 1, text_dim]
merged:       [B, L + 1, text_dim]
```

这就是自驾状态进入模型的主要方式。

### 3.4 Prompt 从机器人任务指令变成驾驶场景描述

NAVSIM prompt 在：

```text
NavSimVideoDataset.build_prompt_fixed()
```

分两种：

```text
use_dynamic_prompt: false
  使用固定描述：
  high-quality photorealistic dashboard camera view...

use_dynamic_prompt: true
  加入历史轨迹、当前速度、加速度稳定性、turn left / straight / turn right 等语义
```

动态 prompt 会把历史 ego trajectory 写成文本，例如：

```text
Over the past 2 seconds, the ego vehicle followed this trajectory: ...
It is currently moving at moderate speed ...
expected to continue turning left ...
```

由于训练时 `load_text_encoder: false`，文本 embedding 需要提前缓存。

相关脚本：

```text
scripts/precompute_navsim_text_embeds.py
```

缓存 key 是 prompt 的 sha256，读取逻辑在：

```text
NavSimVideoDataset._get_cached_text_context()
```

## 4. 核心模型如何构建

NAVSIM 监督训练入口：

```bash
python scripts/train.py task=navsim_uncond_front_384x672_1e-4
```

Shell 多卡入口：

```bash
bash scripts/train_navsim_zero1.sh task=navsim_uncond_front_384x672_1e-4
```

核心调用链：

```text
scripts/train.py
  -> fastwam.runtime.run_training(cfg)
    -> instantiate(cfg.model)
      -> fastwam.runtime.create_fastwam(...)
        -> FastWAM.from_wan22_pretrained(...)
```

`configs/task/navsim_*.yaml` 会覆盖：

```yaml
defaults:
  - override /data: navsim
  - override /model: fastwam_navsim
```

因此模型实际由：

```text
configs/model/fastwam_navsim.yaml
```

定义。

### 4.1 Video expert

Video expert 来自 Wan2.2 TI2V 的 DiT：

```python
components = load_wan22_ti2v_5b_components(...)
video_expert = components.dit
```

配置要点：

```yaml
video_dit_config:
  in_dim: 48
  out_dim: 48
  hidden_dim: 3072
  num_layers: 30
  num_heads: 24
  attn_head_dim: 128
  text_dim: 4096
  video_attention_mask_mode: first_frame_causal
  action_conditioned: false
  action_dim: 3
```

这里 `action_conditioned: false` 很重要：当前 NAVSIM 版不是让 video branch 直接吃 action 条件来生成视频，而是通过 MoT 让 video/action 两个专家一起训练，推理时 action branch 读取第一帧视觉 token。

### 4.2 Action expert

Action expert 是 `ActionDiT`：

```text
src/fastwam/models/wan22/action_dit.py
```

配置要点：

```yaml
action_dit_config:
  action_dim: 3
  hidden_dim: 1024
  ffn_dim: 4096
  num_layers: 30
  num_heads: 24
  attn_head_dim: 128
  text_dim: 4096
```

它对 action trajectory 做 tokenization：

```python
self.action_encoder = nn.Linear(action_dim, hidden_dim)
self.head = nn.Linear(hidden_dim, action_dim)
```

即：

```text
[B, T_action, 3]
  -> Linear(3, hidden_dim)
  -> DiT blocks
  -> Linear(hidden_dim, 3)
```

ActionDiT 的 backbone 可以从 Wan DiT 插值得到：

```text
scripts/preprocess_action_dit_backbone.py
```

NAVSIM 配置使用：

```yaml
action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim_3outdim.pt
```

### 4.3 MoT: video/action 混合层

`MoT` 在：

```text
src/fastwam/models/wan22/mot.py
```

构造时：

```python
mot = MoT(
    mixtures={"video": video_expert, "action": action_expert},
    mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
)
```

MoT 要求 video/action expert 的层数、head 数和 head dim 一致：

```text
num_layers:     都是 30
num_heads:      都是 24
attn_head_dim:  都是 128
```

每一层里，MoT 做的不是简单 concat 后整块 transformer，而是：

1. 分别用 video expert 的当前 block 生成 video Q/K/V。
2. 分别用 action expert 的当前 block 生成 action Q/K/V。
3. 拼接 Q/K/V。
4. 用一个 joint attention mask 做混合 attention。
5. 再把输出切回 video/action 两段。
6. 各自走各自 expert 的 cross-attn、MLP、modulation 和 head。

NAVSIM/FastWAM 当前 attention mask 逻辑在：

```text
FastWAM._build_mot_attention_mask()
```

语义是：

```text
video -> video:
  使用 video expert 自己的 video causal mask。

action -> action:
  action token 之间可以互相 attention。

action -> video:
  action token 只能看第一帧 video token。

video -> action:
  不允许。
```

这正好符合自驾推理需求：预测未来轨迹时，只需要从当前图像提取视觉条件，然后 action denoising 逐步生成未来轨迹。

## 5. 训练过程

训练器是：

```text
src/fastwam/trainer_tensorboard.py
```

训练器会冻结非 DiT 模块：

```python
model.eval()
model.requires_grad_(False)
model.dit.train()
model.dit.requires_grad_(True)
```

这里 `model.dit = model.mot`，所以实际训练的是：

```text
MoT 内的 video expert + action expert
```

如果有 proprio encoder，也会训练：

```text
proprio_encoder
```

VAE 和 text encoder 不训练。

### 5.1 一个 batch 的输入

`FastWAM.build_inputs()` 期望：

```text
video:        [B, 3, T_video, H, W]
action:       [B, T_action, 3]
context:      [B, L, 4096]
context_mask: [B, L]
proprio:      [B, T_action, 8]
```

当前 NAVSIM 默认：

```text
T_video = 9
T_action = 8
```

代码要求：

```text
T_video % 4 == 1
T_action % (T_video - 1) == 0
H, W 是 16 的倍数
```

这也是为什么常用 `num_frames: 9`，因为 Wan VAE 的时间下采样约束要求 `T % 4 == 1`。

### 5.2 loss 计算

核心函数：

```text
FastWAM.training_loss()
```

步骤：

```text
1. VAE 编码 video -> input_latents。
2. 第一帧 latent 作为条件，不加噪。
3. 对 video latents 加 flow matching noise。
4. 对 action trajectory 加 flow matching noise。
5. video_expert.pre_dit(...) 得到 video tokens。
6. action_expert.pre_dit(...) 得到 action tokens。
7. MoT 混合 video/action attention。
8. video_expert.post_dit(...) 预测 video target。
9. action_expert.post_dit(...) 预测 action target。
10. 计算 video MSE loss。
11. 计算 action MSE loss。
12. total loss = lambda_video * loss_video + lambda_action * loss_action。
```

配置中：

```yaml
loss:
  lambda_action: 1.0
```

GRPO 配置中会同时出现：

```yaml
loss:
  lambda_video: 1.0
  lambda_action: 1.0
```

## 6. 推理过程

NAVSIM 评测入口：

```bash
torchrun --nproc_per_node 2 experiments/navsim/eval_navsim.py \
  task=navsim_uncond_front_384x672_1e-4 \
  ckpt=/path/to/step_xxx.pt \
  experiment_name=step_xxx
```

核心文件：

```text
experiments/navsim/eval_navsim.py
configs/sim_navsim.yaml
```

### 6.1 infer_joint

`FastWAM.infer_joint()` 同时生成：

```text
future video
future action trajectory
```

输入：

```text
input_image: 第一帧图像
context/context_mask 或 prompt
proprio: 当前 ego state
action_horizon: 8
num_video_frames: 9
```

它会：

```text
1. 随机初始化 video latents 和 action latents。
2. 把第一帧 image 编码成 first_frame_latents。
3. 固定 video latents 的第 0 帧为 first_frame_latents。
4. 每个 denoising step 同时预测 video/action。
5. 返回 PIL video frames 和 action tensor。
```

适合训练/验证阶段看未来视频质量。

### 6.2 infer_action

`FastWAM.infer_action()` 只生成：

```text
future action trajectory
```

这是 NAVSIM test 默认更合理的模式，因为 benchmark 最终需要的是规划轨迹。

关键优化：

```text
1. 对第一帧 video token 做一次 prefill。
2. 在 MoT 中缓存每层 video K/V。
3. 每个 action denoising step 只重算 action branch。
4. action query attention 到 cached video K/V + current action K/V。
```

相关函数：

```text
MoT.prefill_video_cache()
MoT.forward_action_with_video_cache()
FastWAM._predict_action_noise_with_cache()
```

这比 joint inference 更快，也避免不必要的视频生成。

### 6.3 eval_navsim 的指标

`experiments/navsim/eval_navsim.py` 会把预测 action 反归一化：

```python
pred_action_denorm = dataset.denormalize_action(pred_action)
```

然后计算：

```text
action_l1
action_l2
traj_ade
traj_fde
heading_mae
```

如果推理返回 video，还会计算：

```text
video_psnr
video_ssim
```

预测轨迹会保存为：

```text
evaluate_results/.../pred_actions/{token}.npy
```

## 7. NAVSIM 数据流

数据集配置：

```text
configs/data/navsim.yaml
```

关键字段：

```yaml
navsim_log_path: ${oc.env:NAVSIM_LOG_PATH}
sensor_blobs_path: ${oc.env:NAVSIM_SENSOR_BLOBS_PATH}
scene_filter: ./navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml
split_config_path: ./navsim/navsim/planning/script/config/training/default_train_val_test_log_split.yaml
split_logs_key: train_logs
```

数据读取链路：

```text
NavSimVideoDataset
  -> SceneFilter
  -> SceneLoader
  -> scene token list
  -> scene frames/cameras/ego_status/future_trajectory
```

一个 sample 返回：

```python
{
    "video": video,                 # [3, T, H, W]
    "action": action,               # [8, 3]
    "state": state,                 # [8, 8]
    "proprio": state,               # [8, 8]
    "prompt": prompt,               # str
    "context": context,             # [L, 4096]
    "context_mask": context_mask,   # [L]
    "image_is_pad": ...,
    "action_is_pad": ...,
    "proprio_is_pad": ...,
    "token": token,
}
```

## 8. 当前实现限制

`NavSimVideoDataset` 里有明确 TODO：

```text
当前主要支持 current_plus_future。
history_plus_future 暂未完整支持。
```

原因是 Wan VAE 当前实现把第一帧作为 reference，并且未来帧数量需要满足 4 的倍数相关约束。若要支持“多历史帧 + 未来帧”的自驾规划输入，需要改：

```text
src/fastwam/models/wan22/wan_video_vae.py
```

特别是 `WanVideoVAE.encode_video()` 附近关于历史帧、未来帧、reference frame 的逻辑。

所以当前 NAVSIM 迁移更准确地说是：

```text
single current image + ego state + prompt -> future trajectory
```

而不是完整的：

```text
multi-frame history video -> future trajectory
```

虽然 prompt 中可以包含历史轨迹文本，但视觉输入主要还是当前帧。

## 9. 目录功能

### configs/

Hydra 配置中心。

```text
configs/data/
  数据集配置。navsim.yaml 是自驾数据入口。

configs/model/
  模型结构配置。fastwam_navsim.yaml 是 NAVSIM 监督训练模型。
  fastwam_navsim_grpo.yaml 是当前工作区新增的 GRPO 模型配置。

configs/task/
  具体 task 配置。NAVSIM 有不同分辨率、动态 prompt、GRPO 版本。

configs/train.yaml
  监督训练默认配置。

configs/train_grpo.yaml
  GRPO 训练默认配置。

configs/sim_navsim.yaml
  NAVSIM 评测配置。
```

### scripts/

训练和预处理入口。

```text
train.py
  监督训练入口。

train_grpo.py
  GRPO 训练入口，当前工作区新增。

preprocess_action_dit_backbone.py
  从 Wan DiT 生成 ActionDiT backbone。

precompute_navsim_text_embeds.py
  NAVSIM prompt embedding 缓存。

train_navsim_zero1.sh / train_navsim_zero2.sh
  NAVSIM 多卡训练脚本。

accelerate_configs/
  Accelerate / DeepSpeed 配置。

ds_configs/
  DeepSpeed ZeRO 配置。
```

### src/fastwam/

核心 Python package。

```text
datasets/
  数据读取。
  lerobot/ 是机器人数据。
  navsim/ 是自驾数据。

models/wan22/
  Wan2.2 + FastWAM 模型核心。

runtime.py
  根据 Hydra 配置创建模型、数据集、trainer。

runtime_grpo.py
  GRPO 训练 runtime。

trainer_tensorboard.py
  监督训练器。

trainer_grpo.py
  GRPO 训练器。

utils/
  日志、采样器、视频保存、视频指标、文件系统工具。
```

### experiments/

benchmark 评测和部署。

```text
experiments/libero/
  LIBERO 机器人评测。

experiments/robotwin/
  RoboTwin 机器人评测。

experiments/navsim/
  NAVSIM 自驾评测。
```

### navsim/

NAVSIM devkit / 集成代码副本。

主要包括：

```text
navsim/common/
  dataclass、dataloader、枚举。

navsim/planning/
  PDM scorer、metric cache、simulation、trajectory agent。

navsim/agents/
  baseline / npy trajectory agent。

navsim/visualization/
  BEV、camera、lidar、plot 可视化。

navsim/scripts/
  metric cache、PDM scoring、submission、可视化脚本。
```

### data/

本地数据和缓存。

```text
data/navsim/
  NAVSIM logs、sensor blobs、maps 等，具体依赖环境变量。

data/text_embeds_cache/
  文本 embedding 缓存。

data/metric_cache_navtrain / metric_cache_navtest
  NAVSIM PDM metric cache。

data/libero_mujoco3.3.2/
  原机器人 LIBERO 数据。
```

### runs/

训练输出。

```text
runs/{task}/{run_id}/
  config.yaml
  train log
  checkpoints/
  tensorboard events
```

### evaluate_results/

评测输出。

```text
summary.json
per_sample_results.jsonl
pred_actions/*.npy
videos/*.mp4
```

### checkpoints/

预训练权重和中间 backbone。

```text
Wan / DiffSynth 相关权重
ActionDiT backbone
FastWAM release 或本地 checkpoint
```

### third_party/

第三方 benchmark，目前主要是 RoboTwin。

## 10. 自驾迁移相关提交和当前工作区状态

从当前 git 历史看：

```text
db06b02 Initial commit Raw FastWam
  原始 FastWAM。

28a66fe Add Navsim dataset train FastWAM
  加入 NAVSIM dataset、训练配置、NAVSIM devkit、fastwam_navsim.yaml。

d2dadb7 add fastwam navsim test
  加入 NAVSIM eval、sim_navsim.yaml、npy trajectory agent、评测脚本。

313c654 add visualtion
  加入 NAVSIM 可视化脚本。
```

当前工作区还有未提交改动，主要是：

```text
新增 GRPO:
  configs/model/fastwam_navsim_grpo.yaml
  configs/train_grpo.yaml
  scripts/train_grpo.py
  src/fastwam/models/wan22/fastwam_grpo.py
  src/fastwam/runtime_grpo.py
  src/fastwam/trainer_grpo.py

新增/调整 NAVSIM task:
  navsim_uncond_front_192x352_1e-4.yaml
  navsim_uncond_front_384x672_1e-4.yaml
  navsim_uncond_front_768x1344_1e-4.yaml
  navsim_uncond_front_768x1344_dynpro_1e-4.yaml
  navsim_uncond_grpo_front_192x352_1e-6.yaml

删除旧配置:
  navsim_uncond_front_768x1024_1e-4.yaml
  navsim_uncond_front_768x1024_dynpro_1e-4.yaml
```

## 11. 推荐阅读顺序

理解这个仓库建议按以下顺序读：

```text
1. configs/task/navsim_uncond_front_384x672_1e-4.yaml
2. configs/data/navsim.yaml
3. configs/model/fastwam_navsim.yaml
4. src/fastwam/datasets/navsim/navsim_dataset.py
5. src/fastwam/runtime.py
6. src/fastwam/models/wan22/fastwam.py
7. src/fastwam/models/wan22/action_dit.py
8. src/fastwam/models/wan22/mot.py
9. src/fastwam/trainer_tensorboard.py
10. experiments/navsim/eval_navsim.py
```

如果关注 GRPO，再读：

```text
configs/train_grpo.yaml
configs/model/fastwam_navsim_grpo.yaml
src/fastwam/runtime_grpo.py
src/fastwam/models/wan22/fastwam_grpo.py
src/fastwam/trainer_grpo.py
```

## 12. 一句话总结

FastWAM-XTY 把原来机器人 FastWAM 的动作分支迁移成 NAVSIM ego trajectory 分支，把机器人 proprio 替换为 ego velocity/acceleration/driving command，把机器人多相机观测替换为车载前视图像，并通过 MoT 让轨迹 action token 读取第一帧视觉 token，从而实现：

```text
front camera image + ego state + driving prompt
        -> future ego trajectory
```

训练时仍保留未来视频建模，推理和 NAVSIM 评测时则可以只走 action branch 输出规划轨迹。



```
数据集格式：
 1. RGB 相机图像
      - 代码：src/fastwam/datasets/navsim/navsim_dataset.py
      - camera_layout=front 时只用：
          - cameras.cam_f0.image

      - camera_layout=stitched_front 时拼接：
          - cameras.cam_l0.image
          - cameras.cam_f0.image
          - cameras.cam_r0.image

      - 没有读取 depth image。

  2. 未来 ego trajectory，作为 action label
      - 来自：

        scene.get_future_trajectory(...).poses

      - 形状：

        [future_action_horizon, 3] = [8, 3]

      - 三维含义：

        [x, y, heading]

      - 这是模型预测的核心目标。

  3. 历史 ego trajectory，用来构造动态 prompt
      - 来自：

        scene.get_history_trajectory(...).poses

      - 只在 use_dynamic_prompt=true 时写进文本 prompt。
      - 不是作为数值序列直接喂给模型，而是转成自然语言描述。

  4. ego 状态，作为 proprio 条件
      - 来自：

        scene.get_agent_input().ego_statuses[-1]

      - 使用字段：

        ego_velocity       2 维
        ego_acceleration   2 维
        driving_command    4 维 one-hot

      - 拼成：

        state/proprio: 8 维

      - 经过 Linear(8 -> text_dim) 变成一个额外 context token。

  5. driving command
      - 一方面进入 proprio 的 8 维状态。
      - 另一方面用于动态 prompt，比如 turn left / go straight / turn right。

  6. scene token / split 信息
      - 用 NAVSIM 的 SceneLoader 和 SceneFilter 做数据划分、索引和加载。
      - 配置在 configs/data/navsim.yaml。

  没有用的东西：

  - 没有用深度图。
  - FastWAM dataset 中没有用 LiDAR 点云，而且 SensorConfig 里显式设置了：

    lidar_pc=False

  - 没有把 HD map / lane graph 作为模型输入。
  - 没有把 occupancy、BEV、semantic map 作为 FastWAM 输入。

  需要区分一下：navsim/ 目录里保留了 NAVSIM/TransFuser/PDM 相关代码，其中 TransFuser 配置里能看到 use_depth、LiDAR 处理等逻辑，但这不是当前 FastWAM-NAVSIM
  dataset 使用的路径。当前 FastWAM 的输入路径就是 RGB 图像 + ego 状态 + prompt，监督目标是未来 ego trajectory。GRPO/PDM 相关代码可能会用 NAVSIM metric cache 做
  reward/评测，但也不是把深度图喂进 FastWAM 模型。

```
