import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "npy_trajectory_vis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize selected FastWAM NavSim trajectories.")
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
    parser.add_argument("--token", action="append", default=[], help="Specific token to visualize. Repeat to pass more.")
    parser.add_argument("--token-file", type=Path, default=None, help="Text file containing one token per line.")
    parser.add_argument("--csv-file", type=Path, default=None, help="CSV file containing a 'token' column.")
    parser.add_argument("--csv-filter-column", type=str, default=None, help="Optional CSV filter column.")
    parser.add_argument("--csv-filter-value", type=str, default=None, help="Optional CSV filter value.")
    parser.add_argument("--num-tokens", type=int, default=None, help="Only render the first N selected tokens.")
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


def load_tokens_from_file(token_file: Path) -> list[str]:
    return [line.strip() for line in token_file.read_text().splitlines() if line.strip()]


def load_tokens_from_csv(csv_file: Path, filter_column: str | None, filter_value: str | None) -> list[str]:
    df = pd.read_csv(csv_file)
    if "token" not in df.columns:
        raise KeyError(f"CSV file must contain a 'token' column: {csv_file}")
    if filter_column is not None:
        if filter_column not in df.columns:
            raise KeyError(f"CSV file does not contain filter column '{filter_column}'")
        if filter_value is None:
            raise ValueError("--csv-filter-value is required when --csv-filter-column is set")
        df = df[df[filter_column].astype(str) == filter_value]
    return df["token"].dropna().astype(str).tolist()


def dedupe_tokens(tokens: list[str]) -> list[str]:
    ordered = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def select_tokens(args: argparse.Namespace, scene_loader: SceneLoader) -> list[str]:
    tokens = list(args.token)
    if args.token_file is not None:
        tokens.extend(load_tokens_from_file(args.token_file))
    if args.csv_file is not None:
        tokens.extend(load_tokens_from_csv(args.csv_file, args.csv_filter_column, args.csv_filter_value))

    available_tokens = scene_loader.tokens
    available_token_set = set(available_tokens)
    tokens = dedupe_tokens(tokens)
    if not tokens:
        tokens = available_tokens

    missing = [token for token in tokens if token not in available_token_set]
    if missing:
        print(f"Skip {len(missing)} tokens missing from split/filter: {missing[:10]}")
    tokens = [token for token in tokens if token in available_token_set]

    if args.num_tokens is not None:
        tokens = tokens[: args.num_tokens]
    return tokens


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.agent_config, args.pred_actions_path)
    scene_loader = build_scene_loader(args)
    tokens = select_tokens(args, scene_loader)

    for token in tqdm(tokens, desc="Rendering tokens"):
        scene = scene_loader.get_scene_from_token(token)
        pred_trajectory = agent.compute_trajectory(scene.get_agent_input(), scene)
        fig = render_scene(scene, pred_trajectory, args.plot_mode)
        figure_path = args.output_dir / f"{token}.png"
        fig.savefig(figure_path, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
