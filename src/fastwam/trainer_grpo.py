import json
import lzma
import math
import os
import pickle
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from hydra.utils import instantiate
import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.video_metrics import video_psnr, video_ssim

logger = get_logger(__name__)


class DistributedRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size: int, repeats: int, num_replicas: int, rank: int, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.repeats = int(repeats)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.repeats <= 0:
            raise ValueError("`repeats` must be positive.")
        total = self.batch_size * self.num_replicas
        if total % self.repeats != 0:
            raise ValueError(
                f"global sample batch ({total}) must be divisible by repeats ({self.repeats})."
            )
        self.unique_per_global_batch = total // self.repeats

    def __iter__(self):
        while True:
            gen = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=gen)[: self.unique_per_global_batch].tolist()
            repeated = [idx for idx in indices for _ in range(self.repeats)]
            order = torch.randperm(len(repeated), generator=gen).tolist()
            shuffled = [repeated[i] for i in order]
            start = self.rank * self.batch_size
            yield shuffled[start : start + self.batch_size]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


class NavSimPDMReward:
    DEFAULT_SCORING_CONFIG_PATH = (
        Path(__file__).resolve().parents[2]
        / "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    )

    def __init__(
        self,
        metric_cache_path: str,
        scorer_config: Optional[dict[str, Any]] = None,
        scoring_config_path: Optional[str] = None,
    ):

        self.metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
        scoring_cfg = self._load_scoring_config(scoring_config_path)
        if scorer_config:
            scoring_cfg = OmegaConf.merge(
                scoring_cfg,
                {"scorer": {"config": scorer_config}},
            )
        self.simulator = instantiate(scoring_cfg.simulator)
        self.scorer = instantiate(scoring_cfg.scorer)
        if self.simulator.proposal_sampling != self.scorer.proposal_sampling:
            raise ValueError("PDM simulator and scorer proposal sampling must be identical.")
        self.trajectory_cls = Trajectory
        self.pdm_score = pdm_score
        self._cache: dict[str, Any] = {}

    @classmethod
    def _load_scoring_config(cls, scoring_config_path: Optional[str]) -> DictConfig:
        path = Path(scoring_config_path) if scoring_config_path else cls.DEFAULT_SCORING_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"PDM scoring config not found: {path}")
        cfg = OmegaConf.load(path)
        for key in ("proposal_sampling", "simulator", "scorer"):
            if key not in cfg:
                raise KeyError(f"PDM scoring config `{path}` is missing `{key}`.")
        return cfg

    def _load_metric_cache(self, token: str):
        if token not in self._cache:
            with lzma.open(self.metric_cache_loader.metric_cache_paths[token], "rb") as f:
                self._cache[token] = pickle.load(f)
        return self._cache[token]

    def __call__(self, trajectories: torch.Tensor, tokens: list[str]) -> torch.Tensor:
        pred_np = trajectories.detach().cpu().numpy()
        rewards = []
        for i, token in enumerate(tokens):
            result = self.pdm_score(
                metric_cache=self._load_metric_cache(str(token)),
                model_trajectory=self.trajectory_cls(pred_np[i]),
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
            )
            rewards.append(float(asdict(result)["score"]))
        return torch.tensor(rewards, device=trajectories.device, dtype=trajectories.dtype)


class VideoReward:
    VALID_METRICS = ("psnr", "ssim", "fid", "fvd")

    def __init__(self, metric: str, metric_weights: Optional[dict[str, Any]] = None):
        raw_metric = str(metric).strip().lower()
        parts = [part.strip() for part in re.split(r"[+,]", raw_metric) if part.strip()]
        if not parts:
            raise ValueError(
                "video reward metric must include at least one of: psnr, ssim, fid, fvd."
            )
        invalid = [part for part in parts if part not in self.VALID_METRICS]
        if invalid:
            raise ValueError(
                "video reward metric contains unsupported values "
                f"{invalid}; valid options are: {', '.join(self.VALID_METRICS)}."
            )
        # Keep user order but drop duplicates for stable composite rewards.
        self.metrics = list(dict.fromkeys(parts))
        raw_weights = metric_weights or {}
        if not isinstance(raw_weights, dict):
            raise ValueError("`video_reward_metric_weights` must be a dict when provided.")
        normalized_weights: dict[str, float] = {}
        for key, value in raw_weights.items():
            metric_key = str(key).strip().lower()
            if metric_key not in self.VALID_METRICS:
                raise ValueError(
                    f"Unsupported video reward weight key `{key}`; "
                    f"valid options are: {', '.join(self.VALID_METRICS)}."
                )
            normalized_weights[metric_key] = float(value)
        self.metric_weights = {
            metric_name: normalized_weights.get(metric_name, 1.0) for metric_name in self.metrics
        }
        total_weight = sum(self.metric_weights.values())
        if total_weight <= 0:
            raise ValueError("Sum of `video_reward_metric_weights` for selected metrics must be positive.")
        self.total_weight = float(total_weight)

    @staticmethod
    def _to_01(video: torch.Tensor) -> torch.Tensor:
        return ((video.detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

    @staticmethod
    def _frame_features(video_cthw: torch.Tensor) -> torch.Tensor:
        frames = video_cthw.permute(1, 0, 2, 3)
        pooled = F.adaptive_avg_pool2d(frames, output_size=(16, 16))
        return pooled.flatten(1)

    @staticmethod
    def _frechet_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = y.float()
        mu_x = x.mean(dim=0)
        mu_y = y.mean(dim=0)
        x_centered = x - mu_x
        y_centered = y - mu_y
        denom_x = max(x.shape[0] - 1, 1)
        denom_y = max(y.shape[0] - 1, 1)
        cov_x = x_centered.T @ x_centered / denom_x
        cov_y = y_centered.T @ y_centered / denom_y
        eye = torch.eye(cov_x.shape[0], device=x.device, dtype=x.dtype) * 1e-4
        prod = (cov_x + eye) @ (cov_y + eye)
        eigvals = torch.linalg.eigvals(prod).real.clamp(min=0.0)
        covmean_trace = torch.sqrt(eigvals).sum()
        return (mu_x - mu_y).pow(2).sum() + torch.trace(cov_x + cov_y) - 2.0 * covmean_trace

    def _metric_reward(self, metric: str, pred_i: torch.Tensor, target_i: torch.Tensor) -> float:
        if metric == "psnr":
            return float(video_psnr(pred_i.cpu(), target_i.cpu()) / 50.0)
        if metric == "ssim":
            return float(video_ssim(pred_i.cpu(), target_i.cpu()))
        if metric == "fid":
            return float(
                (-self._frechet_distance(self._frame_features(pred_i), self._frame_features(target_i))).item()
            )
        pred_feat = self._frame_features(pred_i).flatten().unsqueeze(0)
        target_feat = self._frame_features(target_i).flatten().unsqueeze(0)
        return float(-F.mse_loss(pred_feat, target_feat).item())

    def __call__(self, pred_bcthw: torch.Tensor, target_bcthw: torch.Tensor) -> torch.Tensor:
        pred = self._to_01(pred_bcthw)
        target = self._to_01(target_bcthw)
        rewards = []
        for pred_i, target_i in zip(pred, target):
            weighted_reward = 0.0
            for metric in self.metrics:
                weighted_reward += self.metric_weights[metric] * self._metric_reward(metric, pred_i, target_i)
            rewards.append(weighted_reward / self.total_weight)
        return torch.tensor(rewards, device=pred_bcthw.device, dtype=torch.float32)


class FastWAMGRPOTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        del val_dataset
        self.model = model
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.grpo = cfg.grpo
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        self.max_steps = None if cfg.max_steps is None else int(cfg.max_steps)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.seed = int(cfg.seed)
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.max_grad_norm = float(cfg.max_grad_norm)
        if int(self.grpo.sample.train_batch_size) <= 0 or int(self.grpo.train.batch_size) <= 0:
            raise ValueError("GRPO sample/train batch sizes must be positive.")
        if int(self.grpo.sample.num_image_per_prompt) <= 0 or int(self.grpo.sample.sample_time_per_prompt) <= 0:
            raise ValueError("GRPO samples per prompt and sample time per prompt must be positive.")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=(
                int(self.grpo.train.gradient_accumulation_steps) * self._train_steps_per_rollout()
            ),
            mixed_precision=str(cfg.mixed_precision),
            log_with="tensorboard",
            project_dir=self.output_dir,
            step_scheduler_with_optimizer=False,
        )
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            ds_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
            ds_config["train_micro_batch_size_per_gpu"] = int(self.grpo.train.batch_size)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._apply_train_mode(self.model)
        trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]
        if getattr(self.model, "proprio_encoder", None) is not None:
            trainable_params.extend(p for p in self.model.proprio_encoder.parameters() if p.requires_grad)
        if not trainable_params:
            raise ValueError("No trainable parameters found for GRPO training.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self.train_sampler = DistributedRepeatSampler(
            train_dataset,
            batch_size=int(self.grpo.sample.train_batch_size),
            repeats=int(self.grpo.sample.num_image_per_prompt),
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            seed=self.seed,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
        )
        self.video_reward = VideoReward(
            str(self.grpo.video_reward_metric),
            metric_weights=self.grpo.get("video_reward_metric_weights"),
        )
        self.pdm_reward = NavSimPDMReward(
            metric_cache_path=str(self.grpo.metric_cache_path),
            scorer_config=self.grpo.get("pdm_scorer_config"),
            scoring_config_path=self.grpo.get("pdm_scoring_config_path"),
        )
        self.global_step = 0
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        ensure_dir(self.output_dir)
        ensure_dir(self.weights_dir)
        self.accelerator.init_trackers("grpo")
        self._resume_or_load_checkpoint()

    @staticmethod
    def _apply_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        if getattr(model, "use_video_lora", False) or getattr(model, "use_action_lora", False):
            model.activate_lora_training()
        else:
            model.dit.requires_grad_(True)
        if getattr(model, "proprio_encoder", None) is not None:
            model.proprio_encoder.train()
            model.proprio_encoder.requires_grad_(True)

    def _resume_or_load_checkpoint(self):
        resume = self.cfg.get("resume")
        if not resume:
            return
        path = Path(str(resume))
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(path), optimizer=None)
        logger.info("Loaded GRPO initial weights from %s", resume)

    def _train_steps_per_rollout(self):
        return max(
            1,
            int(math.ceil(int(self.grpo.sample.num_steps) * float(self.grpo.train.timestep_fraction))),
        )

    def _save_checkpoint(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            path = os.path.join(self.weights_dir, f"step_{self.global_step:06d}.pt")
            self.accelerator.unwrap_model(self.model).save_checkpoint(path, optimizer=None, step=self.global_step)
            with open(os.path.join(self.checkpoint_root, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump({"global_step": self.global_step}, f, indent=2)
        self.accelerator.wait_for_everyone()

    @staticmethod
    def _tokens_from_batch(batch: dict[str, Any]) -> list[str]:
        tokens = batch.get("token")
        if isinstance(tokens, (list, tuple)):
            return [str(t) for t in tokens]
        return [str(t) for t in tokens]

    def _compute_rewards(self, batch, rollout):
        target_video = batch["video"].to(device=rollout["video"].device, dtype=rollout["video"].dtype)
        video_rewards = self.video_reward(rollout["video"], target_video)
        pred_action = rollout["action"].detach().cpu()
        pred_traj = self.train_dataset.denormalize_action(pred_action).to(rollout["action"].device)
        pdm_rewards = self.pdm_reward(pred_traj, self._tokens_from_batch(batch))
        total = float(self.grpo.video_reward_weight) * video_rewards + float(self.grpo.pdm_reward_weight) * pdm_rewards
        return total.detach(), video_rewards.detach(), pdm_rewards.detach()

    def _sample_epoch(self, epoch: int):
        unwrapped = self.accelerator.unwrap_model(self.model)
        samples = []
        train_iter = iter(self.train_loader)
        for i in range(int(self.grpo.sample.num_batches_per_epoch)):
            self.train_sampler.set_epoch(epoch * int(self.grpo.sample.num_batches_per_epoch) + i)
            batch = next(train_iter)
            for _ in range(int(self.grpo.sample.sample_time_per_prompt)):
                with self.accelerator.autocast():
                    rollout = unwrapped.sample_grpo(
                        batch,
                        num_inference_steps=int(self.grpo.sample.num_steps),
                        tiled=bool(self.grpo.get("tiled", False)),
                        kl_reward=float(self.grpo.sample.get("kl_reward", 0.0)),
                    )
                rewards, video_rewards, pdm_rewards = self._compute_rewards(batch, rollout)
                samples.append(
                    {
                        "video_latents": rollout["video_latents"].detach(),
                        "action_latents": rollout["action_latents"].detach(),
                        "video_log_probs": rollout["video_log_probs"].detach(),
                        "action_log_probs": rollout["action_log_probs"].detach(),
                        "kl": rollout["kl"].detach(),
                        "timesteps_video": rollout["timesteps_video"].detach(),
                        "timesteps_action": rollout["timesteps_action"].detach(),
                        "sigma_video": rollout["sigma_video"].detach(),
                        "sigma_action": rollout["sigma_action"].detach(),
                        "context": rollout["context"].detach(),
                        "context_mask": rollout["context_mask"].detach(),
                        "first_frame_latents": rollout["first_frame_latents"].detach(),
                        "rewards": rewards,
                        "video_rewards": video_rewards,
                        "pdm_rewards": pdm_rewards,
                        "tokens": self._tokens_from_batch(batch),
                    }
                )
        return samples

    def _collate_rollouts(self, samples: list[dict[str, torch.Tensor]]):
        out = {}
        for key in samples[0]:
            if key == "tokens":
                out[key] = [token for s in samples for token in s[key]]
                continue
            if key in {"sigma_video", "sigma_action"}:
                out[key] = samples[0][key]
            else:
                out[key] = torch.cat([s[key] for s in samples], dim=0)
        rewards = out["rewards"].float()
        reward_for_advantage = rewards.unsqueeze(1).expand_as(out["video_log_probs"])
        kl_reward = float(self.grpo.sample.get("kl_reward", 0.0))
        if kl_reward > 0:
            reward_for_advantage = reward_for_advantage - kl_reward * out["kl"].float()
        advantages = self._group_relative_advantages(out["tokens"], reward_for_advantage)
        out["advantages"] = advantages.clamp(-float(self.grpo.train.adv_clip_max), float(self.grpo.train.adv_clip_max))
        out = self._filter_zero_advantage_rollouts(out)
        return out

    def _group_relative_advantages(self, tokens: list[str], rewards: torch.Tensor) -> torch.Tensor:
        original_ndim = rewards.ndim
        local_rewards = rewards.detach().cpu().float()
        if local_rewards.ndim == 1:
            local_rewards = local_rewards.unsqueeze(1)
        local_pairs = [
            (str(token), reward.tolist()) for token, reward in zip(tokens, local_rewards)
        ]
        gathered_pairs = [None for _ in range(self.accelerator.num_processes)]
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(gathered_pairs, local_pairs)
            flat_pairs = [pair for rank_pairs in gathered_pairs for pair in rank_pairs]
        else:
            flat_pairs = local_pairs

        by_token: dict[str, list[list[float]]] = {}
        for token, reward in flat_pairs:
            by_token.setdefault(token, []).append(reward)

        adv = []
        global_rewards = torch.tensor([reward for _, reward in flat_pairs], dtype=torch.float32)
        fallback_mean = global_rewards.mean(dim=0)
        fallback_std = global_rewards.std(dim=0, unbiased=False).clamp(min=1e-4)
        for token, reward in local_pairs:
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            group = torch.tensor(by_token.get(token, [reward]), dtype=torch.float32)
            if group.shape[0] > 1:
                mean = group.mean(dim=0)
                std = group.std(dim=0, unbiased=False).clamp(min=1e-4)
            else:
                mean = fallback_mean
                std = fallback_std
            adv.append((reward_tensor - mean) / std)
        advantages = torch.stack(adv, dim=0).to(device=rewards.device, dtype=rewards.dtype)
        if original_ndim == 1:
            return advantages.squeeze(1)
        return advantages

    def _filter_zero_advantage_rollouts(self, rollouts: dict[str, Any]) -> dict[str, Any]:
        advantages = rollouts["advantages"]
        if advantages.ndim == 1:
            nonzero_mask = advantages.abs() != 0
        else:
            nonzero_mask = advantages.abs().sum(dim=tuple(range(1, advantages.ndim))) != 0
        true_count = int(nonzero_mask.sum().item())
        if true_count == 0:
            rollouts["advantages"] = advantages + 1e-6
            return rollouts

        num_batches = max(1, int(self.grpo.sample.num_batches_per_epoch) * int(self.grpo.sample.sample_time_per_prompt))
        remainder = true_count % num_batches
        if remainder != 0:
            false_indices = torch.where(~nonzero_mask)[0]
            num_to_change = num_batches - remainder
            if false_indices.numel() >= num_to_change:
                selected = false_indices[torch.randperm(false_indices.numel(), device=false_indices.device)[:num_to_change]]
                nonzero_mask[selected] = True

        indices = torch.where(nonzero_mask)[0]
        filtered = {}
        total_batch = advantages.shape[0]
        cpu_indices = indices.detach().cpu().tolist()
        for key, value in rollouts.items():
            if key == "tokens":
                filtered[key] = [value[i] for i in cpu_indices]
            elif isinstance(value, torch.Tensor) and value.shape[:1] == (total_batch,):
                filtered[key] = value[indices]
            else:
                filtered[key] = value
        return filtered

    def _train_rollouts(self, rollouts: dict[str, torch.Tensor]):
        unwrapped = self.accelerator.unwrap_model(self.model)
        self._apply_train_mode(unwrapped)
        total_batch = rollouts["advantages"].shape[0]
        info = {"loss": [], "approx_kl": [], "clipfrac": [], "kl_loss": []}
        for _ in range(int(self.grpo.train.num_inner_epochs)):
            perm = torch.randperm(total_batch, device=self.accelerator.device)
            rollouts = {
                key: (
                    [value[i] for i in perm.detach().cpu().tolist()]
                    if key == "tokens"
                    else value[perm]
                    if isinstance(value, torch.Tensor) and value.shape[:1] == (total_batch,)
                    else value
                )
                for key, value in rollouts.items()
            }
            for start in range(0, total_batch, int(self.grpo.train.batch_size)):
                train_rollouts = self._slice_rollouts(rollouts, start, start + int(self.grpo.train.batch_size))
                for j in range(self._train_steps_per_rollout()):
                    with self.accelerator.accumulate(self.model):
                        with self.accelerator.autocast():
                            outputs = self.model(
                                train_rollouts,
                                j,
                                mode="grpo_log_probs",
                                return_kl=float(self.grpo.train.get("beta", 0.0)) > 0,
                            )
                            kl_beta = float(self.grpo.train.get("beta", 0.0))
                            if kl_beta > 0:
                                logp_video, logp_action, step_kl = outputs
                            else:
                                logp_video, logp_action = outputs
                                step_kl = None
                            old_logp = train_rollouts["video_log_probs"][:, j] + train_rollouts["action_log_probs"][:, j]
                            logp = logp_video + logp_action
                            ratio = torch.exp(logp - old_logp)
                            advantages = train_rollouts["advantages"][:, j]
                            unclipped = -advantages * ratio
                            clip_range = float(self.grpo.train.clip_range)
                            clipped = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                            policy_loss = torch.maximum(unclipped, clipped).mean()
                            if kl_beta > 0 and step_kl is not None:
                                kl_loss = step_kl.mean()
                                loss = policy_loss + kl_beta * kl_loss
                            else:
                                kl_loss = torch.zeros((), device=logp.device, dtype=logp.dtype)
                                loss = policy_loss
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                self.max_grad_norm,
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    info["loss"].append(float(loss.detach().item()))
                    info["approx_kl"].append(float((0.5 * (logp - old_logp).pow(2).mean()).detach().item()))
                    info["clipfrac"].append(
                        float(((ratio - 1.0).abs() > float(self.grpo.train.clip_range)).float().mean().detach().item())
                    )
                    info["kl_loss"].append(float(kl_loss.detach().item()))
        return {k: float(sum(v) / max(len(v), 1)) for k, v in info.items()}

    @staticmethod
    def _slice_rollouts(rollouts: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        out = {}
        for key, value in rollouts.items():
            if key == "tokens":
                out[key] = value[start:end]
            elif isinstance(value, torch.Tensor) and value.shape[:1] == (rollouts["advantages"].shape[0],):
                out[key] = value[start:end]
            else:
                out[key] = value
        return out

    def train(self):
        start = time.perf_counter()
        max_steps = self.max_steps or self.num_epochs
        for epoch in range(self.num_epochs):
            if self.global_step >= max_steps:
                break
            samples = self._sample_epoch(epoch)
            rollouts = self._collate_rollouts(samples)
            train_info = self._train_rollouts(rollouts)
            self.global_step += 1
            metrics = {
                "reward/total": float(rollouts["rewards"].mean().item()),
                "reward/video": float(rollouts["video_rewards"].mean().item()),
                "reward/pdm": float(rollouts["pdm_rewards"].mean().item()),
                "reward/kl": float(rollouts["kl"].mean().item()),
                "train/loss": train_info["loss"],
                "train/approx_kl": train_info["approx_kl"],
                "train/clipfrac": train_info["clipfrac"],
                "train/kl_loss": train_info["kl_loss"],
                "epoch": epoch,
            }
            self.accelerator.log(metrics, step=self.global_step)
            if self.accelerator.is_main_process and self.global_step % self.log_every == 0:
                elapsed = time.perf_counter() - start
                logger.info("step=%d metrics=%s elapsed=%.1fs", self.global_step, metrics, elapsed)
            if self.global_step % self.save_every == 0:
                self._save_checkpoint()
        self._save_checkpoint()
        self.accelerator.end_training()
