import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig, Trajectory
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax, _transform_pcs_to_images
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_CFG = SCRIPT_DIR / "config/common/agent/npy_trajectory_agent.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "npy_trajectory_log_vis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FastWAM NavSim trajectories grouped by log.")
    parser.add_argument("--pred-actions-path", type=Path, required=True, help="Directory containing <token>.npy files.")
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=DEFAULT_AGENT_CFG,
        help="Hydra config for NpyTrajectoryAgent.",
    )
    parser.add_argument("--split", type=str, default="test", help="NavSim split under OPENSCENE_DATA_ROOT/navsim_logs.")
    parser.add_argument("--filter", type=str, default="navtest", help="Scene filter yaml name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered figures.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["bev_camera", "bev"],
        default="bev_camera",
        help="Visualization layout.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Saved image DPI.")
    parser.add_argument("--num-logs", type=int, default=None, help="Only render the first N logs.")
    parser.add_argument(
        "--tokens-per-log",
        type=int,
        default=None,
        help="Only render the first N tokens per log after timestamp sorting.",
    )
    return parser.parse_args()


def build_agent(agent_config_path: Path, pred_actions_path: Path):
    cfg = OmegaConf.load(agent_config_path)
    cfg.pred_actions_path = str(pred_actions_path.resolve())
    agent = instantiate(cfg)
    agent.initialize()
    return agent


def load_scene_filter(filter_name: str) -> SceneFilter:
    filter_cfg_path = SCRIPT_DIR / "config/common/train_test_split/scene_filter" / f"{filter_name}.yaml"
    if not filter_cfg_path.exists():
        raise FileNotFoundError(f"Scene filter config not found: {filter_cfg_path}")
    return instantiate(OmegaConf.load(filter_cfg_path))


def build_sensor_config(plot_mode: str) -> SensorConfig:
    if plot_mode == "bev":
        return SensorConfig.build_no_sensors()
    return SensorConfig(
        cam_f0=True,
        cam_l0=False,
        cam_l1=False,
        cam_l2=False,
        cam_r0=False,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,
    )


def build_scene_loader(args: argparse.Namespace) -> SceneLoader:
    openscene_root_str = os.environ.get("OPENSCENE_DATA_ROOT")
    if not openscene_root_str:
        raise EnvironmentError("OPENSCENE_DATA_ROOT is not set.")
    openscene_root = Path(openscene_root_str).expanduser().resolve()
    scene_filter = load_scene_filter(args.filter)
    return SceneLoader(
        data_path=openscene_root / "navsim_logs" / args.split,
        sensor_blobs_path=openscene_root / "sensor_blobs" / args.split,
        scene_filter=scene_filter,
        sensor_config=build_sensor_config(args.plot_mode),
    )


def render_bev(scene: Scene, pred_trajectory: Trajectory) -> tuple[plt.Figure, plt.Axes]:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    gt_trajectory = scene.get_future_trajectory()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, gt_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, pred_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)
    ax.set_title(scene.scene_metadata.initial_token, fontsize=10)
    fig.tight_layout()
    return fig, ax


def _trajectory_camera_config(color: str) -> dict:
    return {
        "line_color": color,
        "line_color_alpha": 0.8,
        "line_width": 4,
        "line_style": "-",
        "marker": None,
        "zorder": 3,
        "arrow_color": color,
        "arrow_edge_color": color,
        "arrow_alpha": 1.0,
        "arrow_line_width": 1.5,
    }


def _get_intersection_with_image_bottom_boundary(
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    x1, y1 = float(start_pt[0]), float(start_pt[1])
    x2, y2 = float(end_pt[0]), float(end_pt[1])
    if abs(y2 - y1) < 1e-6:
        return None
    target_y = float(image_height - 1)
    t = (target_y - y1) / (y2 - y1)
    if t < 0.0 or t > 1.0:
        return None
    x = x1 + t * (x2 - x1)
    if x < 0 or x > (image_width - 1):
        return None
    return np.array([x, target_y], dtype=np.float32)


def add_trajectory_to_front_camera_ax(ax: plt.Axes, camera, trajectory: Trajectory, config: dict) -> plt.Axes:
    poses_2d = trajectory.poses[:, :2]
    poses_3d = np.concatenate([poses_2d, np.zeros((poses_2d.shape[0], 1), dtype=np.float32)], axis=1)
    poses = np.concatenate([np.array([[0.0, 0.0, 0.0]], dtype=np.float32), poses_3d], axis=0)

    projected_points, in_fov_mask = _transform_pcs_to_images(
        poses.T,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=camera.image.shape[:2],
    )

    image_height, image_width = camera.image.shape[:2]
    points_to_plot = []
    first_point = projected_points[1] if len(projected_points) > 1 else None
    second_point = projected_points[2] if len(projected_points) > 2 else None

    if first_point is not None and in_fov_mask[1]:
        points_to_plot.append(first_point)
    elif first_point is not None and second_point is not None:
        intersection = _get_intersection_with_image_bottom_boundary(
            first_point, second_point, image_width, image_height
        )
        if intersection is not None:
            points_to_plot.append(intersection)

    for idx in range(2, len(projected_points)):
        if in_fov_mask[idx]:
            points_to_plot.append(projected_points[idx])

    if len(points_to_plot) < 2:
        return ax

    plot_points = np.asarray(points_to_plot, dtype=np.float32)
    ax.plot(
        plot_points[:, 0],
        plot_points[:, 1],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config.get("marker"),
        markersize=config.get("marker_size", 0),
        markeredgecolor=config.get("marker_edge_color"),
        zorder=config["zorder"],
    )

    last_point = plot_points[-1]
    second_last_point = plot_points[-2]
    arrow_dx = last_point[0] - second_last_point[0]
    arrow_dy = last_point[1] - second_last_point[1]
    arrow = patches.FancyArrowPatch(
        posA=(last_point[0], last_point[1]),
        posB=(last_point[0] + arrow_dx, last_point[1] + arrow_dy),
        arrowstyle="-|>",
        mutation_scale=15,
        fc=config["arrow_color"],
        ec=config["arrow_edge_color"],
        alpha=config["arrow_alpha"],
        linewidth=config.get("arrow_line_width", config["line_width"]),
        connectionstyle="arc3,rad=0.0",
        zorder=config["zorder"] + 1,
    )
    ax.add_patch(arrow)
    return ax


def render_bev_camera(scene: Scene, pred_trajectory: Trajectory) -> tuple[plt.Figure, object]:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]
    gt_trajectory = scene.get_future_trajectory()

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
    camera_ax = fig.add_subplot(gs[0])
    bev_ax = fig.add_subplot(gs[1])

    add_camera_ax(camera_ax, frame.cameras.cam_f0)
    add_trajectory_to_front_camera_ax(camera_ax, frame.cameras.cam_f0, pred_trajectory, _trajectory_camera_config("red"))
    add_trajectory_to_front_camera_ax(camera_ax, frame.cameras.cam_f0, gt_trajectory, _trajectory_camera_config("green"))

    add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
    add_trajectory_to_bev_ax(bev_ax, gt_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(bev_ax, pred_trajectory, TRAJECTORY_CONFIG["agent"])

    camera_ax.axis("off")
    camera_ax.set_xticks([])
    camera_ax.set_yticks([])
    camera_ax.set_aspect("auto")
    configure_bev_ax(bev_ax)
    configure_ax(bev_ax)
    fig.suptitle(scene.scene_metadata.initial_token, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, left=0.03, right=0.97, top=0.95, bottom=0.03)
    return fig, (camera_ax, bev_ax)


def render_scene(scene: Scene, pred_trajectory: Trajectory, plot_mode: str) -> plt.Figure:
    if plot_mode == "bev":
        fig, _ = render_bev(scene, pred_trajectory)
        return fig
    fig, _ = render_bev_camera(scene, pred_trajectory)
    return fig


def sanitize_log_name(log_name: str) -> str:
    return log_name.replace("/", "_")


def iter_logs(scene_loader: SceneLoader, num_logs: int | None) -> Iterable[tuple[str, list[str]]]:
    tokens_per_log = scene_loader.get_tokens_list_per_log()
    log_names = sorted(tokens_per_log.keys())
    if num_logs is not None:
        log_names = log_names[:num_logs]
    for log_name in log_names:
        token_infos = []
        for token in tokens_per_log[log_name]:
            scene_dict_list = scene_loader.scene_frames_dicts[token]
            center_idx = scene_loader._scene_filter.num_history_frames - 1
            center_frame = scene_dict_list[center_idx]
            token_infos.append((int(center_frame["timestamp"]), token))
        token_infos.sort(key=lambda item: item[0])
        yield log_name, [token for _, token in token_infos]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.agent_config, args.pred_actions_path)
    scene_loader = build_scene_loader(args)

    for log_name, ordered_tokens in iter_logs(scene_loader, args.num_logs):
        if args.tokens_per_log is not None:
            ordered_tokens = ordered_tokens[: args.tokens_per_log]
        log_output_dir = args.output_dir / sanitize_log_name(log_name)
        log_output_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, token in enumerate(tqdm(ordered_tokens, desc=f"Rendering {log_name}")):
            scene = scene_loader.get_scene_from_token(token)
            pred_trajectory = agent.compute_trajectory(scene.get_agent_input(), scene)
            fig = render_scene(scene, pred_trajectory, args.plot_mode)
            figure_path = log_output_dir / f"{frame_idx:05d}_{token}.png"
            fig.savefig(figure_path, bbox_inches="tight", dpi=args.dpi)
            plt.close(fig)


if __name__ == "__main__":
    main()
