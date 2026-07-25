from __future__ import annotations

import copy
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .fastwam import FastWAM
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT


class FastWAMDiT(FastWAM):
    """FastWAM with a pretrained-video-initialized auxiliary depth expert."""

    def __init__(
        self,
        *args,
        flow_expert: WanVideoDiT,
        flow_train_shift: float = 5.0,
        flow_infer_shift: float = 5.0,
        flow_num_train_timesteps: int = 1000,
        loss_lambda_flow: float = 1.0,
        depth_mode: str = "future_denoise",
        depth_mask_ratio: float = 0.4,
        depth_mask_block_size: int = 4,
        loss_lambda_masked_depth: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        valid_depth_modes = {
            "future_denoise",
            "masked_refinement",
            "hybrid",
            "rgb_to_depth_perception",
        }
        if depth_mode not in valid_depth_modes:
            raise ValueError(
                f"`depth_mode` must be one of {sorted(valid_depth_modes)}, got {depth_mode!r}"
            )
        if not 0.0 < depth_mask_ratio <= 1.0:
            raise ValueError(
                f"`depth_mask_ratio` must be in (0, 1], got {depth_mask_ratio}"
            )
        if depth_mask_block_size <= 0:
            raise ValueError(
                f"`depth_mask_block_size` must be positive, got {depth_mask_block_size}"
            )
        self.flow_expert = flow_expert
        self.train_flow_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(flow_num_train_timesteps),
            shift=float(flow_train_shift),
        )
        self.infer_flow_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(flow_num_train_timesteps),
            shift=float(flow_infer_shift),
        )
        self.loss_lambda_flow = float(loss_lambda_flow)
        self.depth_mode = depth_mode
        self.depth_mask_ratio = float(depth_mask_ratio)
        self.depth_mask_block_size = int(depth_mask_block_size)
        self.loss_lambda_masked_depth = float(loss_lambda_masked_depth)
        # This is intentionally zero-initialized rather than randomly initialized.
        # It represents a masked relative-depth latent without adding a new DiT head.
        self.depth_mask_embedding = torch.nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        self.flow_expert.to(device=self.device, dtype=self.torch_dtype)

    @property
    def depth_expert(self) -> WanVideoDiT:
        """Depth uses the same video-VAE/DiT auxiliary pathway as RGB flow."""
        return self.flow_expert

    @property
    def train_depth_scheduler(self) -> WanContinuousFlowMatchScheduler:
        return self.train_flow_scheduler

    @property
    def infer_depth_scheduler(self) -> WanContinuousFlowMatchScheduler:
        return self.infer_flow_scheduler

    @classmethod
    def from_wan22_pretrained(
        cls,
        flow_dit_config: Optional[dict[str, Any]] = None,
        flow_train_shift: float = 5.0,
        flow_infer_shift: float = 5.0,
        flow_num_train_timesteps: int = 1000,
        loss_lambda_flow: float = 1.0,
        depth_mode: str = "future_denoise",
        depth_mask_ratio: float = 0.4,
        depth_mask_block_size: int = 4,
        loss_lambda_masked_depth: float = 1.0,
        **kwargs,
    ):
        base = FastWAM.from_wan22_pretrained(**kwargs)
        video_cfg = dict(kwargs.get("video_dit_config") or {})
        cfg = dict(video_cfg)
        cfg.update(dict(flow_dit_config or {}))
        cfg.setdefault("text_dim", base.text_dim)
        cfg.setdefault("hidden_dim", base.action_expert.hidden_dim)
        cfg.setdefault("ffn_dim", base.action_expert.ffn_dim)
        cfg.setdefault("num_layers", len(base.action_expert.blocks))
        cfg.setdefault("num_heads", base.action_expert.num_heads)
        cfg.setdefault("attn_head_dim", base.action_expert.attn_head_dim)
        cfg.setdefault("freq_dim", base.action_expert.freq_dim)
        cfg.setdefault("has_image_input", False)
        cfg.setdefault("patch_size", getattr(base.video_expert, "patch_size", (1, 2, 2)))
        cfg.setdefault("in_dim", getattr(base.video_expert, "in_dim", 48))
        cfg.setdefault("out_dim", getattr(base.video_expert, "in_dim", 48))
        cfg.setdefault("seperated_timestep", True)
        cfg.setdefault("require_clip_embedding", False)
        cfg.setdefault("require_vae_embedding", False)
        cfg.setdefault("fuse_vae_embedding_in_latents", True)
        cfg.setdefault("video_attention_mask_mode", getattr(base.video_expert, "video_attention_mask_mode", "first_frame_causal"))
        cfg.setdefault("action_conditioned", False)
        if cfg != video_cfg:
            raise ValueError(
                "Depth DiT must exactly match `video_dit_config` so it can be "
                "initialized from the fully pretrained Wan Video DiT."
            )
        flow_expert = copy.deepcopy(base.video_expert)
        mot = MoT(
            mixtures={
                "video": base.video_expert,
                "action": base.action_expert,
                "flow": flow_expert,
            },
            mot_checkpoint_mixed_attn=base.mot.mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=base.video_expert,
            action_expert=base.action_expert,
            mot=mot,
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            text_dim=base.text_dim,
            proprio_dim=base.proprio_dim,
            device=str(base.device),
            torch_dtype=base.torch_dtype,
            video_train_shift=base.train_video_scheduler.shift,
            video_infer_shift=base.infer_video_scheduler.shift,
            video_num_train_timesteps=base.train_video_scheduler.num_train_timesteps,
            action_train_shift=base.train_action_scheduler.shift,
            action_infer_shift=base.infer_action_scheduler.shift,
            action_num_train_timesteps=base.train_action_scheduler.num_train_timesteps,
            loss_lambda_video=base.loss_lambda_video,
            loss_lambda_action=base.loss_lambda_action,
            flow_expert=flow_expert,
            flow_train_shift=flow_train_shift,
            flow_infer_shift=flow_infer_shift,
            flow_num_train_timesteps=flow_num_train_timesteps,
            loss_lambda_flow=loss_lambda_flow,
            depth_mode=depth_mode,
            depth_mask_ratio=depth_mask_ratio,
            depth_mask_block_size=depth_mask_block_size,
            loss_lambda_masked_depth=loss_lambda_masked_depth,
        )
        model.model_paths = getattr(base, "model_paths", {})
        return model

    def training_loss(self, sample, tiled: bool = False):
        if getattr(self, "flow_only_training", False):
            loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
            loss_dict = {}
        else:
            inputs = self.build_inputs(sample, tiled=tiled)
            input_latents = inputs["input_latents"]
            batch_size = input_latents.shape[0]
            context = inputs["context"]
            context_mask = inputs["context_mask"]
            action = inputs["action"]
            action_is_pad = inputs["action_is_pad"]
            image_is_pad = inputs["image_is_pad"]

            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
            if inputs["first_frame_latents"] is not None:
                latents[:, :, 0:1] = inputs["first_frame_latents"]

            noise_action = torch.randn_like(action)
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
            target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

            depth_rgb = sample.get("depth_rgb")
            aux_rgb = depth_rgb if depth_rgb is not None else sample.get("flow_rgb")
            aux_name = "depth" if depth_rgb is not None else "flow"
            flow_pre = None
            target_flow = None
            timestep_flow = None
            depth_mask = None
            if aux_rgb is not None:
                flow_latents = self._encode_aux_video_latents(
                    aux_rgb, name=aux_name, tiled=tiled
                )
                flow_noise = torch.randn_like(flow_latents)
                timestep_flow = self.train_flow_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=flow_latents.dtype,
                )
                if aux_name == "depth" and self.depth_mode == "rgb_to_depth_perception":
                    # GenCeption-style task contract: RGB is the conditioning
                    # modality and no depth latent is supplied to the depth query.
                    # The Wan flow-matching head predicts the velocity from zero
                    # depth latent to the relative-depth latent, so no separate
                    # randomly initialized RGB-to-depth decoder is introduced.
                    timestep_flow = torch.zeros_like(timestep_flow)
                    noisy_flow = torch.zeros_like(flow_latents)
                    target_flow = -flow_latents
                else:
                    noisy_flow = self.train_flow_scheduler.add_noise(
                        flow_latents, flow_noise, timestep_flow
                    )
                    target_flow = self.train_flow_scheduler.training_target(
                        flow_latents, flow_noise, timestep_flow
                    )
                first_aux_frame_latents = None
                if aux_name == "depth" and self.depth_mode != "rgb_to_depth_perception":
                    first_aux_frame_latents = flow_latents[:, :, 0:1].clone()
                    noisy_flow[:, :, 0:1] = first_aux_frame_latents
                    if self.depth_mode in {"masked_refinement", "hybrid"}:
                        depth_mask = self._sample_future_depth_block_mask(flow_latents)
                        noisy_flow = torch.where(
                            depth_mask,
                            self.depth_mask_embedding.to(
                                device=noisy_flow.device, dtype=noisy_flow.dtype
                            ),
                            noisy_flow,
                        )
                flow_pre = self.flow_expert.pre_dit(
                    x=noisy_flow,
                    timestep=timestep_flow,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=bool(
                        getattr(self.depth_expert, "fuse_vae_embedding_in_latents", False)
                    ) if aux_name == "depth" else False,
                )

            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )

            if flow_pre is None:
                attention_mask = self._build_mot_attention_mask(
                    video_seq_len=video_pre["tokens"].shape[1],
                    action_seq_len=action_pre["tokens"].shape[1],
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=video_pre["tokens"].device,
                )
                embeds_all = {
                    "video": video_pre["tokens"],
                    "action": action_pre["tokens"],
                }
                freqs_all = {
                    "video": video_pre["freqs"],
                    "action": action_pre["freqs"],
                }
                context_all = {
                    "video": {
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    "action": {
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                }
                t_mod_all = {
                    "video": video_pre["t_mod"],
                    "action": action_pre["t_mod"],
                }
            else:
                attention_mask = self._build_mot_attention_mask_with_flow(
                    video_seq_len=video_pre["tokens"].shape[1],
                    action_seq_len=action_pre["tokens"].shape[1],
                    flow_seq_len=flow_pre["tokens"].shape[1],
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=video_pre["tokens"].device,
                )
                embeds_all = {
                    "video": video_pre["tokens"],
                    "action": action_pre["tokens"],
                    "flow": flow_pre["tokens"],
                }
                freqs_all = {
                    "video": video_pre["freqs"],
                    "action": action_pre["freqs"],
                    "flow": flow_pre["freqs"],
                }
                context_all = {
                    "video": {
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    "action": {
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                    "flow": {
                        "context": flow_pre["context"],
                        "mask": flow_pre["context_mask"],
                    },
                }
                t_mod_all = {
                    "video": video_pre["t_mod"],
                    "action": action_pre["t_mod"],
                    "flow": flow_pre["t_mod"],
                }

            tokens_out = self.mot(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
            )

            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

            include_initial_video_step = inputs["first_frame_latents"] is None
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video = target_video[:, :, 1:]

            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
            )
            loss_video = (loss_video_per_sample * video_weight).mean()

            action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
            if action_is_pad is not None:
                action_valid = ~action_is_pad
                valid = action_valid.to(device=action_loss_token.device, dtype=action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)

            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
            )
            loss_action = (action_loss_per_sample * action_weight).mean()
            loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
            loss_dict = {
                "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
                "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            }

            is_genception_depth_mode = (
                aux_name == "depth"
                and self.depth_mode == "rgb_to_depth_perception"
            )
            if flow_pre is not None and not is_genception_depth_mode:
                pred_flow = self.flow_expert.post_dit(tokens_out["flow"], flow_pre)
                if aux_name == "depth" and self.depth_mode != "rgb_to_depth_perception":
                    pred_flow = pred_flow[:, :, 1:]
                    target_flow = target_flow[:, :, 1:]
                squared_error = F.mse_loss(
                    pred_flow.float(),
                    target_flow.float(),
                    reduction="none",
                )
                depth_visible = sample.get("depth_visible") if aux_name == "depth" else None
                if depth_visible is None:
                    loss_per_sample = squared_error.mean(dim=(1, 2, 3, 4))
                else:
                    if aux_name == "depth" and self.depth_mode != "rgb_to_depth_perception":
                        depth_visible = depth_visible[:, :, 1:]
                    visible = self._latent_visibility_mask(
                        depth_visible, target_flow, name="depth_visible"
                    )
                    visible = visible.to(dtype=squared_error.dtype)
                    denom = visible.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
                    loss_per_sample = (squared_error * visible).sum(dim=(1, 2, 3, 4)) / denom
                flow_weight = self.train_flow_scheduler.training_weight(timestep_flow).to(
                    device=loss_per_sample.device,
                    dtype=loss_per_sample.dtype,
                )
                loss_flow = (loss_per_sample * flow_weight).mean()
                loss_total = loss_total + self.loss_lambda_flow * loss_flow
                loss_dict[f"loss_{aux_name}"] = self.loss_lambda_flow * float(loss_flow.detach().item())
                if aux_name == "depth" and depth_mask is not None:
                    masked = depth_mask[:, :, 1:].to(dtype=squared_error.dtype)
                    if depth_visible is not None:
                        masked = masked * visible
                    masked_denom = masked.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
                    masked_loss_per_sample = (
                        squared_error * masked
                    ).sum(dim=(1, 2, 3, 4)) / masked_denom
                    loss_masked_depth = (masked_loss_per_sample * flow_weight).mean()
                    if self.depth_mode == "masked_refinement":
                        loss_total = loss_total - self.loss_lambda_flow * loss_flow
                    loss_total = loss_total + self.loss_lambda_masked_depth * loss_masked_depth
                    loss_dict["loss_masked_depth"] = (
                        self.loss_lambda_masked_depth
                        * float(loss_masked_depth.detach().item())
                    )
            if is_genception_depth_mode:
                loss_depth_perception = self._compute_rgb_to_depth_perception_loss(
                    rgb_latents=input_latents,
                    depth_latents=flow_latents,
                    action_pre=action_pre,
                    context=context,
                    context_mask=context_mask,
                    action=action,
                    fuse_vae_embedding_in_latents=inputs[
                        "fuse_vae_embedding_in_latents"
                    ],
                    depth_visible=sample.get("depth_visible"),
                )
                loss_total = loss_total + self.loss_lambda_flow * loss_depth_perception
                loss_dict["loss_depth_perception"] = self.loss_lambda_flow * float(
                    loss_depth_perception.detach().item()
                )
            return loss_total, loss_dict

        aux_name = "depth" if "depth_rgb" in sample else "flow"
        aux_rgb = sample.get("depth_rgb", sample.get("flow_rgb"))
        if aux_rgb is None:
            return loss_total, loss_dict
        if aux_name == "depth" and self.depth_mode == "rgb_to_depth_perception":
            raise ValueError(
                "`rgb_to_depth_perception` requires joint Video/Action training "
                "because its clean RGB-video condition is intentionally not "
                "available in flow-only training."
            )

        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        flow_latents = self._encode_aux_video_latents(aux_rgb, name=aux_name, tiled=tiled)
        batch_size = flow_latents.shape[0]
        noise = torch.randn_like(flow_latents)
        timestep = self.train_flow_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=flow_latents.dtype,
        )
        depth_mask = None
        if aux_name == "depth" and self.depth_mode == "rgb_to_depth_perception":
            timestep = torch.zeros_like(timestep)
            noisy_flow = torch.zeros_like(flow_latents)
            target_flow = -flow_latents
        else:
            noisy_flow = self.train_flow_scheduler.add_noise(flow_latents, noise, timestep)
            target_flow = self.train_flow_scheduler.training_target(flow_latents, noise, timestep)
        if aux_name == "depth" and self.depth_mode != "rgb_to_depth_perception":
            noisy_flow[:, :, 0:1] = flow_latents[:, :, 0:1]
            if self.depth_mode in {"masked_refinement", "hybrid"}:
                depth_mask = self._sample_future_depth_block_mask(flow_latents)
                noisy_flow = torch.where(
                    depth_mask,
                    self.depth_mask_embedding.to(
                        device=noisy_flow.device, dtype=noisy_flow.dtype
                    ),
                    noisy_flow,
                )
        pred_flow = self.flow_expert(
            x=noisy_flow,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.depth_expert, "fuse_vae_embedding_in_latents", False)
            ) if aux_name == "depth" else False,
        )

        if aux_name == "depth" and self.depth_mode != "rgb_to_depth_perception":
            pred_flow = pred_flow[:, :, 1:]
            target_flow = target_flow[:, :, 1:]
        squared_error = F.mse_loss(pred_flow.float(), target_flow.float(), reduction="none")
        depth_visible = sample.get("depth_visible") if aux_name == "depth" else None
        if depth_visible is None:
            loss_per_sample = squared_error.mean(dim=(1, 2, 3, 4))
        else:
            if aux_name == "depth" and self.depth_mode != "rgb_to_depth_perception":
                depth_visible = depth_visible[:, :, 1:]
            visible = self._latent_visibility_mask(depth_visible, target_flow, name="depth_visible")
            visible = visible.to(dtype=squared_error.dtype)
            loss_per_sample = (squared_error * visible).sum(dim=(1, 2, 3, 4)) / visible.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        weight = self.train_flow_scheduler.training_weight(timestep).to(
            device=loss_per_sample.device,
            dtype=loss_per_sample.dtype,
        )
        loss_flow = (loss_per_sample * weight).mean()
        loss_total = loss_total + self.loss_lambda_flow * loss_flow
        loss_dict[f"loss_{aux_name}"] = self.loss_lambda_flow * float(loss_flow.detach().item())
        if aux_name == "depth" and depth_mask is not None:
            masked = depth_mask[:, :, 1:].to(dtype=squared_error.dtype)
            if depth_visible is not None:
                masked = masked * visible
            masked_loss_per_sample = (squared_error * masked).sum(dim=(1, 2, 3, 4))
            masked_loss_per_sample = masked_loss_per_sample / masked.sum(
                dim=(1, 2, 3, 4)
            ).clamp_min(1.0)
            loss_masked_depth = (masked_loss_per_sample * weight).mean()
            if self.depth_mode == "masked_refinement":
                loss_total = loss_total - self.loss_lambda_flow * loss_flow
            loss_total = loss_total + self.loss_lambda_masked_depth * loss_masked_depth
            loss_dict["loss_masked_depth"] = (
                self.loss_lambda_masked_depth * float(loss_masked_depth.detach().item())
            )
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_flow_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        latents_flow: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        timestep_flow: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        flow_pre = self.flow_expert.pre_dit(
            x=latents_flow,
            timestep=timestep_flow,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=False,
        )

        attention_mask = self._build_mot_attention_mask_with_flow(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            flow_seq_len=flow_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
                "flow": flow_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
                "flow": flow_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                "flow": {
                    "context": flow_pre["context"],
                    "mask": flow_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
                "flow": flow_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_flow = self.flow_expert.post_dit(tokens_out["flow"], flow_pre)
        return pred_video, pred_action, pred_flow

    @torch.no_grad()
    def infer_joint_flow(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(
                    f"`action` must have shape [1,T,A] or [T,A], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        flow_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed + 1)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_flow = torch.randn(
            (1, self.vae.model.z_dim, 1, latent_h, latent_w),
            generator=flow_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_flow, infer_deltas_flow = self.infer_flow_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_flow.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action, step_t_flow, step_delta_flow in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
            infer_timesteps_flow,
            infer_deltas_flow,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            timestep_flow = step_t_flow.unsqueeze(0).to(dtype=latents_flow.dtype, device=self.device)
            pred_video, pred_action, pred_flow = self._predict_joint_flow_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                latents_flow=latents_flow,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                timestep_flow=timestep_flow,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_flow = self.infer_flow_scheduler.step(pred_flow, step_delta_flow, latents_flow)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "flow": self._decode_latents(latents_flow, tiled=tiled),
        }

    def _encode_flow_rgb_latents(self, flow_rgb: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        return self._encode_aux_video_latents(flow_rgb, name="flow_rgb", tiled=tiled)

    def _encode_depth_rgb_latents(self, depth_rgb: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        return self._encode_aux_video_latents(depth_rgb, name="depth_rgb", tiled=tiled)

    def _encode_aux_video_latents(
        self,
        aux_rgb: torch.Tensor,
        *,
        name: str,
        tiled: bool = False,
    ) -> torch.Tensor:
        """Encode flow/depth with the exact Wan video VAE used for RGB video."""
        flow_rgb = aux_rgb
        if flow_rgb.ndim == 4:
            flow_rgb = flow_rgb.unsqueeze(2)
        if flow_rgb.ndim != 5:
            raise ValueError(f"`sample['{name}']` must be [B,3,H,W] or [B,3,T,H,W], got {tuple(flow_rgb.shape)}")
        if flow_rgb.shape[1] != 3:
            raise ValueError(f"`sample['{name}']` channel dimension must be 3, got {tuple(flow_rgb.shape)}")
        if name == "depth" and flow_rgb.shape[2] != 9:
            raise ValueError(
                f"`sample['depth_rgb']` must contain current + 8 future frames (T=9), got T={flow_rgb.shape[2]}"
            )
        flow_rgb = flow_rgb.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        return self._encode_video_latents(flow_rgb, tiled=tiled)

    def _compute_rgb_to_depth_perception_loss(
        self,
        *,
        rgb_latents: torch.Tensor,
        depth_latents: torch.Tensor,
        action_pre: dict[str, Any],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        depth_visible: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the strict GenCeption-style clean-RGB to relative-depth path.

        Video and Action keep their normal diffusion losses in the caller.  This
        second MoT pass receives clean RGB-video latents and an all-zero depth
        query, so the depth prediction has no depth-latent input shortcut.
        """
        batch_size = rgb_latents.shape[0]
        zero_timestep = torch.zeros(
            (batch_size,),
            device=rgb_latents.device,
            dtype=rgb_latents.dtype,
        )
        rgb_pre = self.video_expert.pre_dit(
            x=rgb_latents,
            timestep=zero_timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        depth_pre = self.flow_expert.pre_dit(
            x=torch.zeros_like(depth_latents),
            timestep=zero_timestep.to(dtype=depth_latents.dtype),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.depth_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        attention_mask = self._build_mot_attention_mask_with_flow(
            video_seq_len=rgb_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            flow_seq_len=depth_pre["tokens"].shape[1],
            video_tokens_per_frame=int(rgb_pre["meta"]["tokens_per_frame"]),
            device=rgb_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": rgb_pre["tokens"],
                "action": action_pre["tokens"],
                "flow": depth_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": rgb_pre["freqs"],
                "action": action_pre["freqs"],
                "flow": depth_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": rgb_pre["context"],
                    "mask": rgb_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                "flow": {
                    "context": depth_pre["context"],
                    "mask": depth_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": rgb_pre["t_mod"],
                "action": action_pre["t_mod"],
                "flow": depth_pre["t_mod"],
            },
        )
        pred_velocity = self.flow_expert.post_dit(tokens_out["flow"], depth_pre)
        squared_error = F.mse_loss(
            pred_velocity.float(),
            (-depth_latents).float(),
            reduction="none",
        )
        if depth_visible is None:
            return squared_error.mean()
        visible = self._latent_visibility_mask(
            depth_visible,
            depth_latents,
            name="depth_visible",
        ).to(dtype=squared_error.dtype)
        return (squared_error * visible).sum() / visible.sum().clamp_min(1.0)

    def _sample_future_depth_block_mask(self, latents: torch.Tensor) -> torch.Tensor:
        """Sample a spatial block mask on D1:8 at Wan-VAE latent resolution."""
        if latents.ndim != 5:
            raise ValueError(f"`latents` must be [B,C,T,H,W], got {tuple(latents.shape)}")
        batch_size, _, frames, height, width = latents.shape
        mask = torch.zeros(
            (batch_size, 1, frames, height, width),
            device=latents.device,
            dtype=torch.bool,
        )
        if frames <= 1:
            return mask
        block_size = self.depth_mask_block_size
        grid_height = (height + block_size - 1) // block_size
        grid_width = (width + block_size - 1) // block_size
        grid = torch.rand(
            (batch_size, 1, frames - 1, grid_height, grid_width),
            device=latents.device,
        ).lt(self.depth_mask_ratio)
        if not bool(grid.any()):
            grid[0, 0, 0, 0, 0] = True
        future_mask = F.interpolate(
            grid.to(dtype=torch.float32),
            size=(frames - 1, height, width),
            mode="nearest",
        ).to(dtype=torch.bool)
        mask[:, :, 1:] = future_mask
        return mask.expand(-1, latents.shape[1], -1, -1, -1)

    @staticmethod
    def _latent_visibility_mask(
        visibility: torch.Tensor,
        latents: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Downsample pixel visibility to the VAE latent grid for masked denoising."""
        if visibility.ndim == 4:
            visibility = visibility.unsqueeze(1)
        if visibility.ndim != 5:
            raise ValueError(f"`{name}` must be [B,T,H,W] or [B,1,T,H,W], got {tuple(visibility.shape)}")
        if visibility.shape[0] != latents.shape[0]:
            raise ValueError(f"`{name}` batch mismatch: {visibility.shape[0]} vs {latents.shape[0]}")
        visible = F.interpolate(
            visibility.to(dtype=torch.float32),
            size=latents.shape[-3:],
            mode="nearest",
        )
        return visible.expand(-1, latents.shape[1], -1, -1, -1).gt(0.5)

    @torch.no_grad()
    def _build_mot_attention_mask_with_flow(
        self,
        video_seq_len: int,
        action_seq_len: int,
        flow_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len + flow_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        video_slice = slice(0, video_seq_len)
        action_slice = slice(video_seq_len, video_seq_len + action_seq_len)
        flow_slice = slice(video_seq_len + action_seq_len, total_seq_len)
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)

        mask[video_slice, video_slice] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[action_slice, action_slice] = True
        mask[action_slice, :first_frame_tokens] = True

        mask[flow_slice, video_slice] = True
        mask[flow_slice, flow_slice] = True
        mask[flow_slice, action_slice] = True
        return mask
