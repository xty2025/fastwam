from __future__ import annotations

import argparse
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a derived DPFlow RGB cache with background-like global motion removed. "
            "The original cache is never modified."
        )
    )
    parser.add_argument("--src-dir", required=True, help="Source DPFlow RGB cache directory.")
    parser.add_argument("--dst-dir", required=True, help="Destination cache directory.")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files.")
    parser.add_argument("--residual-quantile", type=float, default=0.85)
    parser.add_argument("--min-threshold", type=float, default=1.0)
    parser.add_argument("--dilate-kernel", type=int, default=9)
    parser.add_argument("--copy-meta", action="store_true", help="Copy source _meta.yaml if present.")
    return parser.parse_args()


def build_foreground_mask(
    flow_uv: torch.Tensor,
    valid_mask: torch.Tensor | None,
    residual_quantile: float,
    min_threshold: float,
    dilate_kernel: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if flow_uv.ndim != 3 or flow_uv.shape[0] != 2:
        raise ValueError(f"`flow_uv` must be [2,H,W], got {tuple(flow_uv.shape)}")

    uv = flow_uv.to(dtype=torch.float32)
    h, w = int(uv.shape[-2]), int(uv.shape[-1])
    flat_uv = uv.reshape(2, -1)
    valid_flat = torch.ones(h * w, dtype=torch.bool)
    if valid_mask is not None:
        valid = valid_mask.to(dtype=torch.bool)
        if valid.ndim == 3 and valid.shape[0] == 1:
            valid = valid[0]
        if valid.shape != (h, w):
            raise ValueError(f"`flow_mask` shape mismatch: {tuple(valid.shape)} vs {(h, w)}")
        valid_flat = valid.reshape(-1)

    if int(valid_flat.sum()) > 0:
        median_uv = torch.stack(
            [torch.median(flat_uv[channel, valid_flat]) for channel in range(2)]
        ).view(2, 1, 1)
    else:
        median_uv = torch.zeros(2, 1, 1, dtype=torch.float32)

    residual = uv - median_uv
    residual_mag = torch.linalg.vector_norm(residual, dim=0)
    residual_valid = residual_mag.reshape(-1)[valid_flat]
    if residual_valid.numel() == 0:
        threshold = torch.tensor(float("inf"))
    else:
        threshold = torch.quantile(residual_valid, float(residual_quantile))
        threshold = torch.maximum(threshold, torch.tensor(float(min_threshold)))

    mask = residual_mag >= threshold
    mask &= valid_flat.reshape(h, w)

    if dilate_kernel > 1:
        if dilate_kernel % 2 == 0:
            raise ValueError("`dilate_kernel` must be odd.")
        pad = dilate_kernel // 2
        mask = F.max_pool2d(
            mask.to(dtype=torch.float32)[None, None],
            kernel_size=dilate_kernel,
            stride=1,
            padding=pad,
        )[0, 0].to(dtype=torch.bool)

    stats = {
        "median_u": float(median_uv[0, 0, 0]),
        "median_v": float(median_uv[1, 0, 0]),
        "threshold": float(threshold) if torch.isfinite(threshold) else float("inf"),
        "mask_ratio": float(mask.to(dtype=torch.float32).mean()),
    }
    return mask.unsqueeze(0).contiguous(), stats


def process_one(
    src_path: str,
    dst_dir: str,
    overwrite: bool,
    residual_quantile: float,
    min_threshold: float,
    dilate_kernel: int,
) -> tuple[str, str, float]:
    src = Path(src_path)
    dst = Path(dst_dir) / src.name
    if dst.exists() and not overwrite:
        return ("skipped", src.name, -1.0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")
        payload = torch.load(src, map_location="cpu")
    if "flow_rgb" not in payload:
        raise KeyError(f"Missing `flow_rgb`: {src}")
    if "flow_uv" not in payload:
        raise KeyError(f"Missing `flow_uv`, needed to derive foreground mask: {src}")

    flow_rgb = payload["flow_rgb"]
    if flow_rgb.ndim != 3 or flow_rgb.shape[0] != 3:
        raise ValueError(f"`flow_rgb` must be [3,H,W], got {tuple(flow_rgb.shape)} from {src}")
    if flow_rgb.dtype != torch.uint8:
        flow_rgb = flow_rgb.to(dtype=torch.float32)
        if float(flow_rgb.min()) < -0.5:
            flow_rgb = (flow_rgb.clamp(-1.0, 1.0).add(1.0).mul(127.5)).round()
        flow_rgb = flow_rgb.clamp(0, 255).to(dtype=torch.uint8)

    dynamic_mask, stats = build_foreground_mask(
        flow_uv=payload["flow_uv"],
        valid_mask=payload.get("flow_mask"),
        residual_quantile=residual_quantile,
        min_threshold=min_threshold,
        dilate_kernel=dilate_kernel,
    )
    filtered_rgb = flow_rgb.clone()
    filtered_rgb.masked_fill_(~dynamic_mask.expand_as(filtered_rgb), 0)

    out = {
        "flow_rgb": filtered_rgb.contiguous(),
        "dynamic_mask": dynamic_mask.to(dtype=torch.bool).cpu(),
        "token": payload.get("token"),
        "frame_a": payload.get("frame_a"),
        "frame_b": payload.get("frame_b"),
        "source_cache": str(src),
        "filter_method": "dpflow_rgb_residual_motion_v1",
        "residual_quantile": float(residual_quantile),
        "min_threshold": float(min_threshold),
        "dilate_kernel": int(dilate_kernel),
        **stats,
    }
    for key in ("flow_rgb_ckpt", "video_size", "format_version"):
        if key in payload:
            out[key] = payload[key]

    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    torch.save(out, tmp)
    os.replace(tmp, dst)
    return ("wrote", src.name, stats["mask_ratio"])


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source cache directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.pt"))
    if args.max_samples is not None:
        files = files[: int(args.max_samples)]
    if not files:
        raise ValueError(f"No .pt files found in {src_dir}")

    workers = max(1, int(args.workers))
    wrote = skipped = 0
    mask_sum = 0.0
    mask_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_one,
                str(path),
                str(dst_dir),
                bool(args.overwrite),
                float(args.residual_quantile),
                float(args.min_threshold),
                int(args.dilate_kernel),
            )
            for path in files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="filter-dpflow-rgb"):
            status, _name, mask_ratio = future.result()
            if status == "wrote":
                wrote += 1
                mask_sum += float(mask_ratio)
                mask_count += 1
            elif status == "skipped":
                skipped += 1

    if args.copy_meta and (src_dir / "_meta.yaml").exists():
        shutil.copy2(src_dir / "_meta.yaml", dst_dir / "_source_meta.yaml")

    meta = dst_dir / "_filter_meta.txt"
    avg_mask = mask_sum / max(mask_count, 1)
    meta.write_text(
        "\n".join(
            [
                "format_version=dpflow_rgb_residual_motion_v1",
                f"src_dir={src_dir}",
                f"dst_dir={dst_dir}",
                f"num_inputs={len(files)}",
                f"wrote={wrote}",
                f"skipped={skipped}",
                f"workers={workers}",
                f"residual_quantile={float(args.residual_quantile)}",
                f"min_threshold={float(args.min_threshold)}",
                f"dilate_kernel={int(args.dilate_kernel)}",
                f"avg_mask_ratio_written={avg_mask:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Done. wrote={wrote} skipped={skipped} avg_mask_ratio_written={avg_mask:.4f} dst={dst_dir}")


if __name__ == "__main__":
    main()
