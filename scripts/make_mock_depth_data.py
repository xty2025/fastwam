"""Create a DepthNav-layout fixture: _meta JSONL shards plus flat depth JPGs."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0 : args.height, 0 : args.width]
    records = []
    for sample_index in range(args.samples):
        sample_id = f"mock-{sample_index:04d}"
        for frame_index in range(9):
            rgb_name = f"s/trainval/mock_scene/CAM_F0/{sample_id}-{frame_index:06d}.jpg"
            depth_name = f"{sample_id}-{frame_index:06d}depth.jpg"
            depth_rgb = np.stack(
                [
                    (xx + sample_index * 19) % 256,
                    (yy + frame_index * 23) % 256,
                    (xx + yy + sample_index * 11 + frame_index * 7) % 256,
                ],
                axis=-1,
            ).astype(np.uint8)
            Image.fromarray(depth_rgb).save(args.output / depth_name, quality=95)
            records.append(
                {
                    "img_name": rgb_name,
                    "depth_vis_name": depth_name,
                    "depth_vis_abspath": f"/unavailable/depthnav/{depth_name}",
                    "timestamp": sample_index * 10_000_000 + frame_index * 500_000,
                }
            )
    metadata_dir = args.output / "_meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "navsim_traj_depth_metadata_shard_00").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
