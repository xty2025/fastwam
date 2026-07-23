#!/usr/bin/env python3
"""Audit DepthNav metadata-to-depth/RGB file alignment with a live tqdm view."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from tqdm import tqdm


IMAGE_FIELDS = ("depth_vis_name", "image_path", "depth_image_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-root",
        required=True,
        type=Path,
        help="DepthNav root containing _meta/ or metadata/ and its depth images.",
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=None,
        help="Optional root containing files referenced by metadata img_name.",
    )
    parser.add_argument(
        "--metadata-glob",
        default="*",
        help="Glob under _meta/ or metadata/ to inspect (default: *).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for alignment_summary.json and alignment_errors.csv.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10000,
        help="Maximum detailed error rows to write (default: 10000).",
    )
    return parser.parse_args()


def metadata_dir(depth_root: Path) -> Path:
    for name in ("_meta", "metadata"):
        candidate = depth_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Expected `_meta/` or `metadata/` under {depth_root}")


def iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    payload = path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        for line_number, line in enumerate(payload.splitlines(), start=1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)
        return
    if isinstance(decoded, list):
        for line_number, record in enumerate(decoded, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, record
        return
    if isinstance(decoded, dict):
        records = decoded.get("records", decoded.get("data", [decoded]))
        if not isinstance(records, list):
            raise ValueError(f"{path} must hold a record, list, `records`, or `data`")
        for line_number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, record
        return
    raise ValueError(f"Unsupported JSON root in {path}")


def candidate_paths(root: Path, raw_path: str) -> list[Path]:
    normalized = raw_path.replace("\\", "/").lstrip("/")
    relative = PurePosixPath(normalized)
    candidates = [root / Path(relative)]
    candidates.append(root / relative.name)
    for marker in ("trainval", "train", "val", "test"):
        if marker in relative.parts:
            candidates.append(root / Path(*relative.parts[relative.parts.index(marker) :]))
    return list(dict.fromkeys(candidates))


def resolve_depth_path(depth_root: Path, record: dict[str, Any]) -> Path | None:
    for field in IMAGE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value:
            for candidate in candidate_paths(depth_root, value):
                if candidate.is_file():
                    return candidate
    legacy_path = record.get("depth_vis_abspath")
    if isinstance(legacy_path, str) and legacy_path:
        legacy = Path(legacy_path)
        if legacy.is_file():
            return legacy
        fallback = depth_root / legacy.name
        if fallback.is_file():
            return fallback
    return None


def resolve_rgb_path(rgb_root: Path, image_name: str) -> Path | None:
    for candidate in candidate_paths(rgb_root, image_name):
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    depth_root = args.depth_root.expanduser().resolve()
    rgb_root = args.rgb_root.expanduser().resolve() if args.rgb_root else None
    meta_root = metadata_dir(depth_root)
    metadata_files = sorted(path for path in meta_root.glob(args.metadata_glob) if path.is_file())
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files match {args.metadata_glob!r} in {meta_root}")

    report_dir = (args.report_dir or depth_root / "alignment_audit").expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    source_to_depth: dict[str, set[str]] = defaultdict(set)
    counters = defaultdict(int)

    progress = tqdm(metadata_files, desc="metadata", unit="shard", dynamic_ncols=True)
    for metadata_file in progress:
        try:
            records = iter_records(metadata_file)
            for line_number, record in records:
                counters["records"] += 1
                source = record.get("img_name", record.get("image_path"))
                depth_path = resolve_depth_path(depth_root, record)
                rgb_path = None
                if not isinstance(source, str) or not source:
                    counters["missing_source_key"] += 1
                    if len(errors) < args.max_errors:
                        errors.append({"kind": "missing_source_key", "metadata": str(metadata_file), "line": str(line_number), "source": "", "depth": ""})
                else:
                    if depth_path is not None:
                        source_to_depth[source].add(str(depth_path))
                    if rgb_root is not None:
                        rgb_path = resolve_rgb_path(rgb_root, source)
                        if rgb_path is None:
                            counters["missing_rgb"] += 1
                            if len(errors) < args.max_errors:
                                errors.append({"kind": "missing_rgb", "metadata": str(metadata_file), "line": str(line_number), "source": source, "depth": str(depth_path or "")})
                if depth_path is None:
                    counters["missing_depth"] += 1
                    if len(errors) < args.max_errors:
                        errors.append({"kind": "missing_depth", "metadata": str(metadata_file), "line": str(line_number), "source": str(source or ""), "depth": ""})
                progress.set_postfix(
                    rows=counters["records"],
                    depth_missing=counters["missing_depth"],
                    rgb_missing=counters["missing_rgb"],
                    bad_source=counters["missing_source_key"],
                )
        except Exception as error:
            counters["bad_metadata_files"] += 1
            if len(errors) < args.max_errors:
                errors.append({"kind": "bad_metadata", "metadata": str(metadata_file), "line": "", "source": "", "depth": str(error)})

    ambiguous_sources = {source: paths for source, paths in source_to_depth.items() if len(paths) > 1}
    counters["ambiguous_source_to_depth"] = len(ambiguous_sources)
    for source, paths in list(ambiguous_sources.items())[: args.max_errors - len(errors)]:
        errors.append({"kind": "ambiguous_source_to_depth", "metadata": "", "line": "", "source": source, "depth": " | ".join(sorted(paths))})

    summary = {
        "depth_root": str(depth_root),
        "rgb_root": str(rgb_root) if rgb_root else None,
        "metadata_dir": str(meta_root),
        "metadata_files": len(metadata_files),
        "unique_rgb_sources": len(source_to_depth),
        **dict(sorted(counters.items())),
        "status": "PASS" if not errors else "FAIL",
    }
    (report_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (report_dir / "alignment_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "metadata", "line", "source", "depth"])
        writer.writeheader()
        writer.writerows(errors)

    print("\nDepthNav alignment summary")
    for key, value in summary.items():
        print(f"{key:30} {value}")
    print(f"reports: {report_dir}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
