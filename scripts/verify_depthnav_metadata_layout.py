"""Verify strict ``img_name -> depth_vis_name`` alignment for a DepthNav root."""

from argparse import ArgumentParser
from pathlib import Path

import torch

from fastwam.datasets.depth_sequence import MetadataDepthSequence


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--sample-id", default="mock-0000")
    args = parser.parse_args()

    loader = MetadataDepthSequence(
        str(args.depth_root),
        image_size=(64, 96),
        alignment_mode="source_path",
        timestamp_tolerance=0.0,
    )
    frame_indices = list(range(9))
    depth_rgb, depth_visible = loader.load(
        navsim_tokens=[f"unused-{frame_index}" for frame_index in frame_indices],
        navsim_timestamps=[frame_index * 500_000 for frame_index in frame_indices],
        navsim_image_paths=[
            f"s/trainval/mock_scene/CAM_F0/{args.sample_id}-{frame_index:06d}.jpg"
            for frame_index in frame_indices
        ],
    )
    assert depth_rgb.shape == (3, 9, 64, 96)
    assert depth_visible.shape == (9, 64, 96)
    assert depth_visible.dtype == torch.bool

    try:
        loader.load(
            navsim_tokens=["unused"] * 9,
            navsim_timestamps=[frame_index * 500_000 for frame_index in frame_indices],
            navsim_image_paths=[f"missing/{frame_index:06d}.jpg" for frame_index in frame_indices],
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Source-path alignment accepted a missing RGB path.")

    print("DEPTHNAV_SOURCE_PATH_ALIGNMENT_OK", tuple(depth_rgb.shape), tuple(depth_visible.shape))


if __name__ == "__main__":
    main()
