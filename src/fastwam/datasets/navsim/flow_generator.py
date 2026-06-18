from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class BEVFlowConfig:
    bev_size: tuple[int, int] = (200, 200)
    resolution: float = 0.5
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    horizon_index: int = 0
    current_frame_index: int | None = None
    include_agents: bool = True


@dataclass(frozen=True)
class BEVFlowTarget:
    flow_gt: torch.Tensor
    valid_mask: torch.Tensor
    agent_mask: torch.Tensor
    static_mask: torch.Tensor
    agent_count: int
    matched_agent_count: int


def generate_navsim_bev_flow(scene: Any, future_trajectory: torch.Tensor, cfg: BEVFlowConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate MOTUS-style BEV flow from NavSim ego motion plus optional tracked boxes.

    Output flow is [2, H, W] in meters. Channel 0 is forward-x displacement and
    channel 1 is left-y displacement from current ego frame to the selected
    future ego frame. The mask is [1, H, W] and marks supervised BEV cells.
    """
    target = generate_navsim_bev_flow_target(scene=scene, future_trajectory=future_trajectory, cfg=cfg)
    return target.flow_gt.contiguous(), target.valid_mask.contiguous()


def generate_navsim_bev_flow_target(scene: Any, future_trajectory: torch.Tensor, cfg: BEVFlowConfig) -> BEVFlowTarget:
    """Generate a dense BEV motion-flow target plus MOTUS-style region masks."""
    flow = _ego_background_flow(future_trajectory=future_trajectory, cfg=cfg)
    valid_mask = torch.ones((1, int(cfg.bev_size[0]), int(cfg.bev_size[1])), dtype=torch.bool)
    agent_mask = torch.zeros_like(valid_mask)
    agent_count = 0
    matched_agent_count = 0

    if cfg.include_agents:
        agent_mask_2d, agent_count, matched_agent_count = _try_apply_agent_overrides(flow, scene, cfg)
        agent_mask[0] = agent_mask_2d
    static_mask = valid_mask & ~agent_mask

    return BEVFlowTarget(
        flow_gt=flow.contiguous(),
        valid_mask=valid_mask.contiguous(),
        agent_mask=agent_mask.contiguous(),
        static_mask=static_mask.contiguous(),
        agent_count=int(agent_count),
        matched_agent_count=int(matched_agent_count),
    )


def _ego_background_flow(future_trajectory: torch.Tensor, cfg: BEVFlowConfig) -> torch.Tensor:
    if future_trajectory.ndim != 2 or future_trajectory.shape[-1] < 3:
        raise ValueError(
            "`future_trajectory` must be [T, >=3] with x/y/yaw in ego frame, "
            f"got {tuple(future_trajectory.shape)}"
        )
    horizon = min(max(int(cfg.horizon_index), 0), int(future_trajectory.shape[0]) - 1)
    ego_delta = future_trajectory[horizon, :3].detach().to(dtype=torch.float32, device="cpu")
    dx, dy, yaw = float(ego_delta[0]), float(ego_delta[1]), float(ego_delta[2])

    xs, ys = _bev_meter_grid(cfg)
    cos_yaw = float(np.cos(-yaw))
    sin_yaw = float(np.sin(-yaw))
    rel_x = xs - dx
    rel_y = ys - dy
    next_x = cos_yaw * rel_x - sin_yaw * rel_y
    next_y = sin_yaw * rel_x + cos_yaw * rel_y
    return torch.stack([next_x - xs, next_y - ys], dim=0).to(dtype=torch.float32)


def _bev_meter_grid(cfg: BEVFlowConfig) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = int(cfg.bev_size[0]), int(cfg.bev_size[1])
    resolution = float(cfg.resolution)
    if cfg.x_range is None:
        x_min, x_max = 0.0, h * resolution
    else:
        x_min, x_max = cfg.x_range
    if cfg.y_range is None:
        half_width = 0.5 * w * resolution
        y_min, y_max = -half_width, half_width
    else:
        y_min, y_max = cfg.y_range

    x_centers = torch.linspace(float(x_min), float(x_max), h + 1)[:-1] + (float(x_max) - float(x_min)) / h * 0.5
    y_centers = torch.linspace(float(y_min), float(y_max), w + 1)[:-1] + (float(y_max) - float(y_min)) / w * 0.5
    xs, ys = torch.meshgrid(x_centers, y_centers, indexing="ij")
    return xs, ys


def _try_apply_agent_overrides(flow: torch.Tensor, scene: Any, cfg: BEVFlowConfig) -> tuple[torch.Tensor, int, int]:
    agent_mask = torch.zeros(flow.shape[-2:], dtype=torch.bool, device=flow.device)
    frames = getattr(scene, "frames", None)
    if not frames:
        return agent_mask, 0, 0
    current_frame_idx = _current_frame_index(scene, cfg)
    next_frame_idx = min(current_frame_idx + int(cfg.horizon_index) + 1, len(frames) - 1)
    if current_frame_idx < 0 or next_frame_idx <= current_frame_idx:
        return agent_mask, 0, 0

    current_boxes = _frame_box_map(frames[current_frame_idx])
    next_boxes = _frame_box_map(frames[next_frame_idx])
    if not current_boxes or not next_boxes:
        return agent_mask, len(current_boxes), 0

    xs, ys = _bev_meter_grid(cfg)
    xs = xs.to(device=flow.device, dtype=flow.dtype)
    ys = ys.to(device=flow.device, dtype=flow.dtype)
    matched_agent_count = 0

    for track_id, box_t in current_boxes.items():
        box_next = next_boxes.get(track_id)
        if box_next is None:
            continue
        center_t = _box_center_xy(box_t)
        center_next = _box_center_xy(box_next)
        dims = _box_length_width(box_t)
        yaw = _box_yaw(box_t)
        yaw_next = _box_yaw(box_next)
        if center_t is None or center_next is None or dims is None:
            continue
        yaw_t = 0.0 if yaw is None else float(yaw)
        yaw_n = yaw_t if yaw_next is None else float(yaw_next)
        polygon = _box_polygon(center=center_t, length_width=dims, yaw=yaw_t)
        override_mask = _rasterize_polygon(polygon, cfg)
        if not override_mask.any():
            continue
        override_mask = override_mask.to(device=flow.device)
        matched_agent_count += 1

        center_t_tensor = torch.tensor(center_t, dtype=flow.dtype, device=flow.device)
        center_next_tensor = torch.tensor(center_next, dtype=flow.dtype, device=flow.device)
        rel_x = xs[override_mask] - center_t_tensor[0]
        rel_y = ys[override_mask] - center_t_tensor[1]
        delta_yaw = _normalize_angle(yaw_n - yaw_t)
        cos_yaw = float(np.cos(delta_yaw))
        sin_yaw = float(np.sin(delta_yaw))
        next_x = center_next_tensor[0] + cos_yaw * rel_x - sin_yaw * rel_y
        next_y = center_next_tensor[1] + sin_yaw * rel_x + cos_yaw * rel_y
        flow[0, override_mask] = next_x - xs[override_mask]
        flow[1, override_mask] = next_y - ys[override_mask]
        agent_mask |= override_mask

    return agent_mask, len(current_boxes), matched_agent_count


def _normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _current_frame_index(scene: Any, cfg: BEVFlowConfig) -> int:
    if cfg.current_frame_index is not None:
        return int(cfg.current_frame_index)
    scene_filter = getattr(scene, "scene_filter", None)
    if scene_filter is not None and getattr(scene_filter, "num_history_frames", None) is not None:
        return int(scene_filter.num_history_frames) - 1
    frames = getattr(scene, "frames", None)
    return max(len(frames) // 2, 0) if frames else -1


def _frame_box_map(frame: Any) -> dict[str, Any]:
    annotations = getattr(frame, "annotations", None)
    if annotations is not None and hasattr(annotations, "boxes") and hasattr(annotations, "track_tokens"):
        names = getattr(annotations, "names", None)
        box_map = {}
        for idx, (box, track_token) in enumerate(zip(annotations.boxes, annotations.track_tokens)):
            if track_token is None:
                continue
            if names is not None and idx < len(names) and str(names[idx]) != "vehicle":
                continue
            box_map[str(track_token)] = np.asarray(box, dtype=np.float32)
        return box_map

    boxes = {}
    for candidate in _iter_frame_annotations(frame):
        track_id = _track_id(candidate)
        if track_id is not None:
            boxes[str(track_id)] = candidate
    return boxes


def _iter_frame_annotations(frame: Any) -> Iterable[Any]:
    containers = [
        "annotations",
        "anns",
        "boxes",
        "agent_boxes",
        "tracked_objects",
        "detections",
        "objects",
    ]
    for name in containers:
        value = getattr(frame, name, None)
        if value is None:
            continue
        if hasattr(value, "tracked_objects"):
            value = value.tracked_objects
        if isinstance(value, dict):
            yield from value.values()
        else:
            try:
                yield from value
            except TypeError:
                continue


def _track_id(box: Any) -> Optional[str]:
    for name in ("track_token", "track_id", "token", "id", "instance_token"):
        value = getattr(box, name, None)
        if value is not None:
            return str(value)
    return None


def _box_center_xy(box: Any) -> Optional[np.ndarray]:
    if isinstance(box, np.ndarray):
        arr = box.reshape(-1)
        if arr.shape[0] >= 2:
            return arr[:2].astype(np.float32)
    for name in ("center", "translation", "position"):
        value = getattr(box, name, None)
        xy = _xy_from_value(value)
        if xy is not None:
            return xy
    oriented_box = getattr(box, "box", None) or getattr(box, "oriented_box", None)
    if oriented_box is not None:
        return _box_center_xy(oriented_box)
    return None


def _xy_from_value(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        return np.asarray([float(value.x), float(value.y)], dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 2:
        return None
    return arr[:2]


def _box_length_width(box: Any) -> Optional[tuple[float, float]]:
    if isinstance(box, np.ndarray):
        arr = box.reshape(-1)
        if arr.shape[0] >= 5:
            return float(arr[3]), float(arr[4])
    names = [
        ("length", "width"),
        ("size_x", "size_y"),
        ("extent_x", "extent_y"),
    ]
    for length_name, width_name in names:
        length = getattr(box, length_name, None)
        width = getattr(box, width_name, None)
        if length is not None and width is not None:
            return float(length), float(width)
    size = getattr(box, "size", None) or getattr(box, "dimensions", None)
    if size is not None:
        arr = np.asarray(size, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 2:
            return float(arr[0]), float(arr[1])
    oriented_box = getattr(box, "box", None) or getattr(box, "oriented_box", None)
    if oriented_box is not None:
        return _box_length_width(oriented_box)
    return None


def _box_yaw(box: Any) -> Optional[float]:
    if isinstance(box, np.ndarray):
        arr = box.reshape(-1)
        if arr.shape[0] >= 7:
            return float(arr[6])
    for name in ("yaw", "heading", "orientation"):
        value = getattr(box, name, None)
        if value is None:
            continue
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        if hasattr(value, "yaw_pitch_roll"):
            return float(value.yaw_pitch_roll[0])
    oriented_box = getattr(box, "box", None) or getattr(box, "oriented_box", None)
    if oriented_box is not None:
        return _box_yaw(oriented_box)
    return None


def _box_polygon(center: np.ndarray, length_width: tuple[float, float], yaw: float) -> np.ndarray:
    length, width = length_width
    local = np.asarray(
        [
            [0.5 * length, 0.5 * width],
            [0.5 * length, -0.5 * width],
            [-0.5 * length, -0.5 * width],
            [-0.5 * length, 0.5 * width],
        ],
        dtype=np.float32,
    )
    rot = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float32,
    )
    return local @ rot.T + center.reshape(1, 2)


def _rasterize_polygon(polygon_xy: np.ndarray, cfg: BEVFlowConfig) -> torch.Tensor:
    xs, ys = _bev_meter_grid(cfg)
    poly = torch.as_tensor(polygon_xy, dtype=torch.float32)
    inside = torch.zeros_like(xs, dtype=torch.bool)
    j = poly.shape[0] - 1
    for i in range(poly.shape[0]):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        crosses = ((yi > ys) != (yj > ys)) & (
            xs < (xj - xi) * (ys - yi) / (yj - yi + 1.0e-6) + xi
        )
        inside ^= crosses
        j = i
    return inside
