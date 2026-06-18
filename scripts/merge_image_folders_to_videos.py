#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable, List

import cv2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge images from each subfolder under a root directory into one MP4 video per subfolder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory whose immediate subdirectories contain image frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store merged videos. Default: <input-dir>_videos",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find image folders instead of only using immediate subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    return parser.parse_args()


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def iter_image_folders(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for folder in sorted(path for path in root.rglob("*") if path.is_dir()):
            if list_image_files(folder):
                yield folder
        return

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if list_image_files(folder):
            yield folder


def ensure_even_size(width: int, height: int) -> tuple[int, int]:
    even_width = width if width % 2 == 0 else width - 1
    even_height = height if height % 2 == 0 else height - 1
    if even_width <= 0 or even_height <= 0:
        raise ValueError(f"Invalid frame size after enforcing even dimensions: {width}x{height}")
    return even_width, even_height


def build_output_path(folder: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_folder = folder.relative_to(input_dir)
    if relative_folder == Path("."):
        name = input_dir.name
    else:
        name = str(relative_folder).replace("/", "__")
    return output_dir / f"{name}.mp4"


def write_video_from_folder(folder: Path, input_dir: Path, output_dir: Path, fps: float, overwrite: bool) -> Path | None:
    image_files = list_image_files(folder)
    if not image_files:
        return None

    output_path = build_output_path(folder, input_dir, output_dir)
    if output_path.exists() and not overwrite:
        print(f"skip existing: {output_path}")
        return output_path

    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first image: {image_files[0]}")

    height, width = first_frame.shape[:2]
    width, height = ensure_even_size(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    try:
        for image_path in image_files:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"skip unreadable image: {image_path}")
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            else:
                frame = frame[:height, :width]

            writer.write(frame)
    finally:
        writer.release()

    return output_path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_dir.parent / f"{input_dir.name}_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = list(iter_image_folders(input_dir, args.recursive))
    if not folders:
        raise RuntimeError(f"No image folders found under: {input_dir}")

    print(f"Found {len(folders)} image folder(s) under {input_dir}")
    print(f"Videos will be written to {output_dir}")

    written = 0
    for index, folder in enumerate(folders, start=1):
        output_path = write_video_from_folder(folder, input_dir, output_dir, args.fps, args.overwrite)
        if output_path is not None:
            written += 1
            print(f"[{index}/{len(folders)}] wrote {output_path}")

    print(f"Done. Wrote {written} video(s).")


if __name__ == "__main__":
    main()
