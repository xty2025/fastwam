from contextlib import ExitStack, contextmanager
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAMGRPO(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.use_video_lora = False
        self.use_action_lora = False
        self.video_lora_path: str | None = None
        self.action_lora_path: str | None = None

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        checkpoint_path: str | None = None,
        use_video_lora: bool = False,
        use_action_lora: bool = False,
        video_lora_path: str | None = None,
        action_lora_path: str | None = None,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        logger.info(f"Loading Wan2.2-TI2V-5B (DiT VAE and text encoder) pretrained weights and configs for FastWAM initialization...")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        if checkpoint_path:
            model.load_model_checkpoint(checkpoint_path, optimizer=None)
        if use_video_lora or use_action_lora:
            model.enable_lora(
                use_video_lora=use_video_lora,
                use_action_lora=use_action_lora,
                video_lora_path=video_lora_path,
                action_lora_path=action_lora_path,
            )
        return model

    @staticmethod
    def _lora_target_modules() -> list[str]:
        return ["q", "k", "v", "o"]

    def _wrap_expert_with_lora(self, expert, lora_path: str | None = None, r: int = 32, lora_alpha: int = 64):
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "`peft` is required when `use_video_lora=true` or `use_action_lora=true`."
            ) from exc

        if lora_path:
            wrapped = PeftModel.from_pretrained(expert, lora_path)
            wrapped.set_adapter("default")
            return wrapped
        lora_config = LoraConfig(
            r=int(r),
            lora_alpha=int(lora_alpha),
            init_lora_weights="gaussian",
            target_modules=self._lora_target_modules(),
        )
        return get_peft_model(expert, lora_config)

    def enable_lora(
        self,
        use_video_lora: bool = True,
        use_action_lora: bool = True,
        video_lora_path: str | None = None,
        action_lora_path: str | None = None,
        r: int = 32,
        lora_alpha: int = 64,
    ) -> None:
        if not use_video_lora and not use_action_lora:
            raise ValueError("At least one of `use_video_lora` or `use_action_lora` must be true.")

        self.mot.requires_grad_(False)
        if use_video_lora:
            self.video_expert = self._wrap_expert_with_lora(
                self.video_expert,
                lora_path=video_lora_path,
                r=r,
                lora_alpha=lora_alpha,
            )
            self.mot.mixtures["video"] = self.video_expert
        if use_action_lora:
            self.action_expert = self._wrap_expert_with_lora(
                self.action_expert,
                lora_path=action_lora_path,
                r=r,
                lora_alpha=lora_alpha,
            )
            self.mot.mixtures["action"] = self.action_expert

        self.dit = self.mot
        self.use_video_lora = bool(use_video_lora)
        self.use_action_lora = bool(use_action_lora)
        self.video_lora_path = video_lora_path
        self.action_lora_path = action_lora_path
        self.video_expert.to(self.device)
        self.action_expert.to(self.device)
        self.activate_lora_training()

    def activate_lora_training(self) -> None:
        if not (self.use_video_lora or self.use_action_lora):
            return
        self.mot.train()
        self.mot.requires_grad_(False)
        if self.use_video_lora:
            self.video_expert.train()
            if hasattr(self.video_expert, "set_adapter"):
                self.video_expert.set_adapter("default")
            for name, param in self.video_expert.named_parameters():
                param.requires_grad_("lora_" in name)
        if self.use_action_lora:
            self.action_expert.train()
            if hasattr(self.action_expert, "set_adapter"):
                self.action_expert.set_adapter("default")
            for name, param in self.action_expert.named_parameters():
                param.requires_grad_("lora_" in name)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def decode_latents_to_tensor(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return video_tensor.detach().float().clamp(-1, 1)

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        action_dim = int(action.shape[2])
        
        # TODO and NOTE: since the VAE encoder use the first frame as reference, and the future frames must be 4 times;
        # if using history 4 frame as reference in autonomous trajectory prediction, the action horizon must be divisible by 
        #  (num_frames - 4 - 1) which is the number of video transitions after the first 4 history frames.
        
        # if action_dim == 3 or action_dim == 2: ## for autonomous traj
        #     if action_horizon % (num_frames - 4 - 1) != 0 or action_horizon % (num_frames - 1) != 0:  ## the first 4 frame are reference frame
        #         raise ValueError(
        #             f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 4 - 1}), got {action_horizon}"
        #         )
        # else:
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        
        ## encode the video latents
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        
        ## the first frame as the condition for training, which is not noised.
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
                
            ## the first state send to the proprio encoder as the condition with the text tokens 
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        # prepare inputs
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        
        ## get the sample time
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        
        ## the first frame latents as the condations for training, which is not noised. 
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

        ## forward pass through experts and MoT
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

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        ## build MoT attention mask (action can attend to the first frame of video, but not the other frames; video cannot attend to action)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
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
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
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

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
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
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
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
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    def _predict_joint_velocity_trainable(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return (
            self.video_expert.post_dit(tokens_out["video"], video_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    @staticmethod
    def _sde_step_with_logprob(
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        sigma_max: torch.Tensor,
        sigma_min: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        logprob_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_output = model_output.float()
        sample = sample.float()
        sigma = sigma.to(sample.device, dtype=torch.float32).view(-1, *([1] * (sample.ndim - 1)))
        sigma_prev = sigma_prev.to(sample.device, dtype=torch.float32).view(-1, *([1] * (sample.ndim - 1)))
        sigma_max = sigma_max.to(sample.device, dtype=torch.float32)
        sigma_min = sigma_min.to(sample.device, dtype=torch.float32)
        dt = sigma_prev - sigma
        sigma_safe = sigma.clamp(min=1e-6)
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
        transition_std = (std_dev_t * torch.sqrt((-dt).clamp(min=1e-12))).clamp(min=1e-6)
        prev_sample_mean = (
            sample * (1 + std_dev_t.pow(2) / (2 * sigma_safe) * dt)
            + model_output * (1 + std_dev_t.pow(2) * (1 - sigma) / (2 * sigma_safe)) * dt
        )
        if deterministic:
            prev_sample = sample + model_output * dt
        elif prev_sample is None:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + transition_std * noise
        else:
            prev_sample = prev_sample.float()

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * transition_std.pow(2))
            - torch.log(transition_std)
            - torch.log(torch.sqrt(torch.as_tensor(2.0 * torch.pi, device=sample.device)))
        )
        if logprob_mask is not None:
            mask = logprob_mask.to(device=log_prob.device, dtype=log_prob.dtype)
            log_prob = log_prob * mask
            denom = mask.expand_as(log_prob).sum(dim=tuple(range(1, log_prob.ndim))).clamp(min=1.0)
            log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim))) / denom
        else:
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        prev_sample = prev_sample.to(dtype=sample.dtype)
        if return_stats:
            return prev_sample, log_prob, prev_sample_mean, transition_std
        return prev_sample, log_prob

    @staticmethod
    def _video_logprob_mask(latents_video: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(latents_video, dtype=torch.float32)
        mask[:, :, 0:1] = 0
        return mask

    @staticmethod
    def _kl_from_sde_means(
        prev_sample_mean: torch.Tensor,
        ref_prev_sample_mean: torch.Tensor,
        transition_std: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kl = (prev_sample_mean.float() - ref_prev_sample_mean.float()).pow(2)
        kl = kl / (2.0 * transition_std.float().pow(2).clamp(min=1e-12))
        if mask is not None:
            mask = mask.to(device=kl.device, dtype=kl.dtype)
            kl = kl * mask
            denom = mask.expand_as(kl).sum(dim=tuple(range(1, kl.ndim))).clamp(min=1.0)
            return kl.sum(dim=tuple(range(1, kl.ndim))) / denom
        return kl.mean(dim=tuple(range(1, kl.ndim)))

    @contextmanager
    def _disable_lora_adapters(self):
        with ExitStack() as stack:
            for module in (self.video_expert, self.action_expert):
                disable_adapter = getattr(module, "disable_adapter", None)
                if callable(disable_adapter):
                    stack.enter_context(disable_adapter())
            yield

    def _prepare_grpo_context(self, sample: dict[str, torch.Tensor], tiled: bool = False) -> dict[str, torch.Tensor]:
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        proprio = sample.get("proprio")
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            proprio0 = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio0)
        input_image = video[:, :, 0]
        first_frame_latents = self.vae.encode(
            [frame.unsqueeze(1) for frame in input_image],
            device=self.device,
            tiled=tiled,
        )
        if isinstance(first_frame_latents, list):
            first_frame_latents = torch.stack(first_frame_latents, dim=0)
        return {
            "context": context,
            "context_mask": context_mask,
            "first_frame_latents": first_frame_latents.to(device=self.device, dtype=self.torch_dtype),
            "num_frames": torch.tensor(video.shape[2], device=self.device),
            "height": torch.tensor(video.shape[3], device=self.device),
            "width": torch.tensor(video.shape[4], device=self.device),
            "action_horizon": torch.tensor(sample["action"].shape[1], device=self.device),
            "gt_action": sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
        }

    def sample_grpo(
        self,
        sample: dict[str, torch.Tensor],
        num_inference_steps: int,
        sigma_shift: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        tiled: bool = False,
        kl_reward: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            cond = self._prepare_grpo_context(sample, tiled=tiled)
            batch_size = cond["context"].shape[0]
            num_frames = int(cond["num_frames"].item())
            height = int(cond["height"].item())
            width = int(cond["width"].item())
            action_horizon = int(cond["action_horizon"].item())
            latent_t = (num_frames - 1) // self.vae.temporal_downsample_factor + 1
            latent_h = height // self.vae.upsampling_factor
            latent_w = width // self.vae.upsampling_factor
            latents_video = torch.randn(
                (batch_size, self.vae.model.z_dim, latent_t, latent_h, latent_w),
                generator=generator,
                device=self.device,
                dtype=self.torch_dtype,
            )
            latents_action = torch.randn(
                (batch_size, action_horizon, self.action_expert.action_dim),
                generator=generator,
                device=self.device,
                dtype=self.torch_dtype,
            )
            first_frame_latents = cond["first_frame_latents"]
            latents_video[:, :, 0:1] = first_frame_latents
            fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
            video_logprob_mask = self._video_logprob_mask(latents_video)

            timesteps_video, _ = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps, self.device, latents_video.dtype, sigma_shift
            )
            timesteps_action, _ = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps, self.device, latents_action.dtype, sigma_shift
            )
            sigma_video = timesteps_video.float() / float(self.infer_video_scheduler.num_train_timesteps)
            sigma_action = timesteps_action.float() / float(self.infer_action_scheduler.num_train_timesteps)
            sigma_video_steps = torch.cat([sigma_video, sigma_video.new_zeros(1)])
            sigma_action_steps = torch.cat([sigma_action, sigma_action.new_zeros(1)])

            video_chain = [latents_video.float()]
            action_chain = [latents_action.float()]
            video_log_probs = []
            action_log_probs = []
            kl_rewards = []
            for i, (step_t_video, step_t_action) in enumerate(zip(timesteps_video, timesteps_action)):
                current_video = latents_video
                current_action = latents_action
                timestep_video = step_t_video.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
                timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
                pred_video, pred_action = self._predict_joint_velocity_trainable(
                    latents_video=current_video,
                    latents_action=current_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=cond["context"],
                    context_mask=cond["context_mask"],
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=None,
                )
                latents_video, logp_video, video_mean, video_std = self._sde_step_with_logprob(
                    pred_video,
                    current_video,
                    sigma_video_steps[i].expand(batch_size),
                    sigma_video_steps[i + 1].expand(batch_size),
                    sigma_video_steps[0],
                    sigma_video_steps[-1],
                    generator=generator,
                    deterministic=deterministic,
                    logprob_mask=video_logprob_mask,
                    return_stats=True,
                )
                latents_action, logp_action, action_mean, action_std = self._sde_step_with_logprob(
                    pred_action,
                    current_action,
                    sigma_action_steps[i].expand(batch_size),
                    sigma_action_steps[i + 1].expand(batch_size),
                    sigma_action_steps[0],
                    sigma_action_steps[-1],
                    generator=generator,
                    deterministic=deterministic,
                    return_stats=True,
                )
                if kl_reward > 0:
                    with self._disable_lora_adapters():
                        ref_pred_video, ref_pred_action = self._predict_joint_velocity_trainable(
                            latents_video=current_video,
                            latents_action=current_action,
                            timestep_video=timestep_video,
                            timestep_action=timestep_action,
                            context=cond["context"],
                            context_mask=cond["context_mask"],
                            fuse_vae_embedding_in_latents=fuse_flag,
                            gt_action=None,
                        )
                    _, _, ref_video_mean, _ = self._sde_step_with_logprob(
                        ref_pred_video,
                        current_video,
                        sigma_video_steps[i].expand(batch_size),
                        sigma_video_steps[i + 1].expand(batch_size),
                        sigma_video_steps[0],
                        sigma_video_steps[-1],
                        prev_sample=latents_video,
                        logprob_mask=video_logprob_mask,
                        return_stats=True,
                    )
                    _, _, ref_action_mean, _ = self._sde_step_with_logprob(
                        ref_pred_action,
                        current_action,
                        sigma_action_steps[i].expand(batch_size),
                        sigma_action_steps[i + 1].expand(batch_size),
                        sigma_action_steps[0],
                        sigma_action_steps[-1],
                        prev_sample=latents_action,
                        return_stats=True,
                    )
                    kl_video = self._kl_from_sde_means(video_mean, ref_video_mean, video_std, mask=video_logprob_mask)
                    kl_action = self._kl_from_sde_means(action_mean, ref_action_mean, action_std)
                    kl_rewards.append(kl_video + kl_action)
                else:
                    kl_rewards.append(torch.zeros(batch_size, device=self.device, dtype=torch.float32))
                latents_video = latents_video.to(dtype=self.torch_dtype)
                latents_action = latents_action.to(dtype=self.torch_dtype)
                latents_video[:, :, 0:1] = first_frame_latents
                video_chain.append(latents_video.float())
                action_chain.append(latents_action.float())
                video_log_probs.append(logp_video)
                action_log_probs.append(logp_action)

            decoded_video = self.decode_latents_to_tensor(latents_video, tiled=tiled)
            return {
                "video": decoded_video,
                "action": latents_action.float(),
                "video_latents": torch.stack(video_chain, dim=1),
                "action_latents": torch.stack(action_chain, dim=1),
                "video_log_probs": torch.stack(video_log_probs, dim=1),
                "action_log_probs": torch.stack(action_log_probs, dim=1),
                "kl": torch.stack(kl_rewards, dim=1),
                "timesteps_video": timesteps_video.float().unsqueeze(0).expand(batch_size, -1),
                "timesteps_action": timesteps_action.float().unsqueeze(0).expand(batch_size, -1),
                "sigma_video": sigma_video_steps.float(),
                "sigma_action": sigma_action_steps.float(),
                "context": cond["context"].detach(),
                "context_mask": cond["context_mask"].detach(),
                "first_frame_latents": first_frame_latents.detach(),
            }

    def grpo_log_probs(
        self,
        rollout: dict[str, torch.Tensor],
        step_index: int,
        return_kl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = rollout["context"].shape[0]
        latents_video = rollout["video_latents"][:, step_index].to(device=self.device, dtype=self.torch_dtype)
        next_video = rollout["video_latents"][:, step_index + 1].to(device=self.device, dtype=self.torch_dtype)
        latents_action = rollout["action_latents"][:, step_index].to(device=self.device, dtype=self.torch_dtype)
        next_action = rollout["action_latents"][:, step_index + 1].to(device=self.device, dtype=self.torch_dtype)
        pred_video, pred_action = self._predict_joint_velocity_trainable(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=rollout["timesteps_video"][:, step_index].to(device=self.device, dtype=self.torch_dtype),
            timestep_action=rollout["timesteps_action"][:, step_index].to(device=self.device, dtype=self.torch_dtype),
            context=rollout["context"].to(device=self.device, dtype=self.torch_dtype),
            context_mask=rollout["context_mask"].to(device=self.device, dtype=torch.bool),
            fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
            gt_action=None,
        )
        sigma_video = rollout["sigma_video"].to(device=self.device)
        sigma_action = rollout["sigma_action"].to(device=self.device)
        video_logprob_mask = self._video_logprob_mask(latents_video)
        _, logp_video, video_mean, video_std = self._sde_step_with_logprob(
            pred_video,
            latents_video,
            sigma_video[step_index].expand(batch_size),
            sigma_video[step_index + 1].expand(batch_size),
            sigma_video[0],
            sigma_video[-1],
            prev_sample=next_video,
            logprob_mask=video_logprob_mask,
            return_stats=True,
        )
        _, logp_action, action_mean, action_std = self._sde_step_with_logprob(
            pred_action,
            latents_action,
            sigma_action[step_index].expand(batch_size),
            sigma_action[step_index + 1].expand(batch_size),
            sigma_action[0],
            sigma_action[-1],
            prev_sample=next_action,
            return_stats=True,
        )
        if return_kl:
            with torch.no_grad():
                with self._disable_lora_adapters():
                    ref_pred_video, ref_pred_action = self._predict_joint_velocity_trainable(
                        latents_video=latents_video,
                        latents_action=latents_action,
                        timestep_video=rollout["timesteps_video"][:, step_index].to(
                            device=self.device, dtype=self.torch_dtype
                        ),
                        timestep_action=rollout["timesteps_action"][:, step_index].to(
                            device=self.device, dtype=self.torch_dtype
                        ),
                        context=rollout["context"].to(device=self.device, dtype=self.torch_dtype),
                        context_mask=rollout["context_mask"].to(device=self.device, dtype=torch.bool),
                        fuse_vae_embedding_in_latents=bool(
                            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
                        ),
                        gt_action=None,
                    )
                _, _, ref_video_mean, _ = self._sde_step_with_logprob(
                    ref_pred_video,
                    latents_video,
                    sigma_video[step_index].expand(batch_size),
                    sigma_video[step_index + 1].expand(batch_size),
                    sigma_video[0],
                    sigma_video[-1],
                    prev_sample=next_video,
                    logprob_mask=video_logprob_mask,
                    return_stats=True,
                )
                _, _, ref_action_mean, _ = self._sde_step_with_logprob(
                    ref_pred_action,
                    latents_action,
                    sigma_action[step_index].expand(batch_size),
                    sigma_action[step_index + 1].expand(batch_size),
                    sigma_action[0],
                    sigma_action[-1],
                    prev_sample=next_action,
                    return_stats=True,
                )
            kl_video = self._kl_from_sde_means(video_mean, ref_video_mean, video_std, mask=video_logprob_mask)
            kl_action = self._kl_from_sde_means(action_mean, ref_action_mean, action_std)
            return logp_video, logp_action, kl_video + kl_action
        return logp_video, logp_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
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
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
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
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
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
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
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

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
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

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

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
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
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
        '''
        Infer action sequence from a single image and optional prompt/context.
        '''
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
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

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        # use the first frame as the condition for action inference, which is not noised.
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

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

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_model_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        if "mot" in payload:
            incompatible = self.mot.load_state_dict(payload["mot"], strict=False)
            missing_keys.extend(list(incompatible.missing_keys))
            unexpected_keys.extend(list(incompatible.unexpected_keys))
        elif "dit" in payload and hasattr(self, "dit"):
            incompatible = self.dit.load_state_dict(payload["dit"], strict=False)
            missing_keys.extend(list(incompatible.missing_keys))
            unexpected_keys.extend(list(incompatible.unexpected_keys))
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            incompatible = self.video_expert.load_state_dict(payload["dit"], strict=False)
            missing_keys.extend(list(incompatible.missing_keys))
            unexpected_keys.extend(list(incompatible.unexpected_keys))
        else:
            raise ValueError(f"Checkpoint missing supported model weights (`mot` or `dit`): {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                incompatible = self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                missing_keys.extend(list(incompatible.missing_keys))
                unexpected_keys.extend(list(incompatible.unexpected_keys))
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        logger.info(
            "Loaded checkpoint: %s | payload_keys=%s | step=%s | missing_keys=%s | unexpected_keys=%s",
            path,
            sorted(payload.keys()),
            payload.get("step"),
            missing_keys,
            unexpected_keys,
        )
        return payload

    def load_checkpoint(self, path, optimizer=None):
        return self.load_model_checkpoint(path, optimizer=optimizer)

    def forward(self, *args, **kwargs):
        mode = kwargs.pop("mode", None)
        if mode == "grpo_log_probs":
            return self.grpo_log_probs(*args, **kwargs)
        return self.training_loss(*args, **kwargs)
