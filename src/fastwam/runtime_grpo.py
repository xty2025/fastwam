import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
)
from .trainer_grpo import FastWAMGRPOTrainer
from .utils import misc
from .utils.logging_config import setup_logging


def create_fastwam_grpo(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    checkpoint_path: str | None = None,
    use_video_lora: bool = False,
    use_action_lora: bool = False,
    video_lora_path: str | None = None,
    action_lora_path: str | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_grpo import FastWAMGRPO

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    action_dit_config = action_dit_config or {}
    video_scheduler = video_scheduler or {}
    action_scheduler = action_scheduler or {}
    loss = loss or {}

    return FastWAMGRPO.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        checkpoint_path=checkpoint_path,
        use_video_lora=bool(use_video_lora),
        use_action_lora=bool(use_action_lora),
        video_lora_path=video_lora_path,
        action_lora_path=action_lora_path,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def run_grpo_training(cfg: DictConfig):
    misc.register_work_dir(cfg.output_dir)
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
        log_file=Path(cfg.output_dir) / "train_grpo.log",
    )
    os.makedirs(str(cfg.output_dir), exist_ok=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f)

    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(mixed_precision),
        device=_resolve_train_device(),
    )
    train_ds, val_ds = build_datasets(cfg.data)
    trainer = FastWAMGRPOTrainer(cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds)
    trainer.train()
