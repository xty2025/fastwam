from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize original and filtered DPFlow RGB cache entries.")
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--filtered-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def to_pil_chw_uint8(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu()
    if array.dtype != torch.uint8:
        array = array.to(dtype=torch.float32)
        if float(array.min()) < -0.5:
            array = (array.clamp(-1, 1).add(1).mul(127.5)).round()
        array = array.clamp(0, 255).to(dtype=torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def label(image: Image.Image, text: str) -> Image.Image:
    pad = 22
    out = Image.new("RGB", (image.width, image.height + pad), "white")
    out.paste(image, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), text, fill=(0, 0, 0))
    return out


def hstack(images: list[Image.Image]) -> Image.Image:
    width = sum(img.width for img in images)
    height = max(img.height for img in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width
    return out


def vstack(images: list[Image.Image]) -> Image.Image:
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height
    return out


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir)
    filtered_dir = Path(args.filtered_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob("*.pt"))
    pairs = [(path, filtered_dir / path.name) for path in src_files if (filtered_dir / path.name).exists()]
    if not pairs:
        raise ValueError(f"No matching cache pairs found: {src_dir} -> {filtered_dir}")
    rng = random.Random(int(args.seed))
    selected = rng.sample(pairs, k=min(int(args.num_samples), len(pairs)))

    rows: list[Image.Image] = []
    report_lines: list[str] = []
    for idx, (src_path, filtered_path) in enumerate(selected):
        src = torch.load(src_path, map_location="cpu")
        filtered = torch.load(filtered_path, map_location="cpu")
        original_rgb = to_pil_chw_uint8(src["flow_rgb"])
        filtered_rgb = to_pil_chw_uint8(filtered["flow_rgb"])
        mask = filtered["dynamic_mask"].to(dtype=torch.bool)[0].numpy()
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")

        overlay = original_rgb.copy()
        red = Image.new("RGB", original_rgb.size, (255, 0, 0))
        alpha = Image.fromarray((mask.astype(np.uint8) * 105), mode="L")
        overlay.paste(red, (0, 0), alpha)

        token = str(filtered.get("token") or src.get("token") or src_path.name)[:16]
        mask_ratio = float(filtered.get("mask_ratio", mask.mean()))
        row = hstack(
            [
                label(original_rgb, f"{idx:02d} original {token}"),
                label(mask_img, f"mask ratio={mask_ratio:.3f}"),
                label(filtered_rgb, "filtered rgb"),
                label(overlay, "mask overlay"),
            ]
        )
        sample_path = out_dir / f"sample_{idx:02d}_{token}.jpg"
        row.save(sample_path, quality=92)
        rows.append(row)
        report_lines.append(
            f"{sample_path.name}: token={token}, mask_ratio={mask_ratio:.4f}, "
            f"threshold={float(filtered.get('threshold', -1.0)):.4f}, "
            f"median_uv=({float(filtered.get('median_u', 0.0)):.4f}, {float(filtered.get('median_v', 0.0)):.4f})"
        )

    contact = vstack(rows)
    contact.save(out_dir / "contact_sheet.jpg", quality=92)
    (out_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote visualization to {out_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
