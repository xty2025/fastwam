from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from fastwam.datasets.navsim.flow_generator import BEVFlowConfig, generate_navsim_bev_flow_target
from fastwam.datasets.navsim.navsim_dataset import NavSimVideoDataset
from fastwam.utils.config_resolvers import register_default_resolvers
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute NavSim MOTUS-style BEV flow cache.")
    parser.add_argument("--config-name", default="train", help="Hydra root config name.")
    parser.add_argument("--task", default="navsim_uncond_front_192x352_xty_dit", help="Task config to compose.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Dataset split to precompute.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_default_resolvers()
    overrides = [f"task={args.task}", "data.train.generate_flow_gt=true", *args.overrides]
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)

    data_cfg = cfg.data.train if args.split == "train" or cfg.data.get("val") is None else cfg.data.val
    cache_dir = data_cfg.get("flow_cache_dir")
    if cache_dir is None:
        raise ValueError("`flow_cache_dir` must be set to precompute flow cache.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene_filter = NavSimVideoDataset._resolve_scene_filter(data_cfg.scene_filter)
    scene_filter = NavSimVideoDataset._apply_split_log_filter(
        scene_filter=scene_filter,
        split_config_path=str(data_cfg.split_config_path),
        split_logs_key=str(data_cfg.split_logs_key),
    )
    scene_loader = SceneLoader(
        data_path=Path(str(data_cfg.navsim_log_path)),
        sensor_blobs_path=Path(str(data_cfg.sensor_blobs_path)),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = list(scene_loader.tokens)
    total = len(tokens) if args.max_samples is None else min(len(tokens), int(args.max_samples))
    flow_cfg = BEVFlowConfig(
        bev_size=(int(data_cfg.flow_bev_size[0]), int(data_cfg.flow_bev_size[1])),
        resolution=float(data_cfg.flow_resolution),
        x_range=None if data_cfg.get("flow_x_range") is None else (float(data_cfg.flow_x_range[0]), float(data_cfg.flow_x_range[1])),
        y_range=None if data_cfg.get("flow_y_range") is None else (float(data_cfg.flow_y_range[0]), float(data_cfg.flow_y_range[1])),
        horizon_index=int(data_cfg.flow_horizon_index),
        current_frame_index=int(scene_filter.num_history_frames) - 1,
        include_agents=bool(data_cfg.flow_include_agents),
    )

    for idx in tqdm(range(total), desc=f"precompute-flow/{args.split}"):
        token = tokens[idx]
        cache_path = _flow_cache_path(cache_dir, token, flow_cfg)
        if cache_path.exists():
            continue
        scene = scene_loader.get_scene_from_token(token)
        future_trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=int(data_cfg.future_action_horizon)).poses,
            dtype=torch.float32,
        )
        target = generate_navsim_bev_flow_target(
            scene=scene,
            future_trajectory=future_trajectory,
            cfg=flow_cfg,
        )
        torch.save(
            {
                "flow_gt": target.flow_gt.to(dtype=torch.float32).cpu(),
                "flow_mask": target.valid_mask.to(dtype=torch.bool).cpu(),
                "valid_mask": target.valid_mask.to(dtype=torch.bool).cpu(),
                "agent_mask": target.agent_mask.to(dtype=torch.bool).cpu(),
                "static_mask": target.static_mask.to(dtype=torch.bool).cpu(),
                "token": str(token),
                "bev_size": tuple(flow_cfg.bev_size),
                "resolution": float(flow_cfg.resolution),
                "x_range": flow_cfg.x_range,
                "y_range": flow_cfg.y_range,
                "horizon_index": int(flow_cfg.horizon_index),
                "include_agents": bool(flow_cfg.include_agents),
                "agent_count": int(target.agent_count),
                "matched_agent_count": int(target.matched_agent_count),
                "format_version": "motus_rigid_v1",
            },
            cache_path,
        )

    meta_path = cache_dir / "_meta.yaml"
    OmegaConf.save(
        {
            "task": args.task,
            "split": args.split,
            "num_samples": total,
            "format_version": "motus_rigid_v1",
            "data": OmegaConf.to_container(cfg.data, resolve=True),
        },
        meta_path,
    )
    print(f"Wrote {total} flow cache entries to {cache_dir}")


def _flow_cache_path(cache_dir: Path, token: str, flow_cfg: BEVFlowConfig) -> Path:
    safe_token = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
    h, w = flow_cfg.bev_size
    name = f"{safe_token}.bev{h}x{w}_res{flow_cfg.resolution:g}_h{flow_cfg.horizon_index}.pt"
    return cache_dir / name


if __name__ == "__main__":
    main()
