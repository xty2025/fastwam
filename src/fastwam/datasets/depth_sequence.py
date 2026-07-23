"""Strict metadata-driven relative-depth loading for FastWAM."""

from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class MetadataDepthSequence:
    """Load a depth sequence by matching every NavSim frame to metadata JSON.

    Supports both generic ``metadata/*.json + images/*.png`` and the provided
    DepthNav layout: ``_meta/*`` JSONL shards plus root-level ``*depth.jpg``.
    DepthNav records are aligned from ``img_name`` to ``depth_vis_name`` or
    ``depth_vis_abspath`` using the original NavSim CAM_F0 data path.
    """

    def __init__(
        self,
        dataset_root: str,
        *,
        image_size: tuple[int, int],
        num_frames: int = 9,
        metadata_glob: str = "*",
        alignment_mode: str = "source_path_then_token_then_timestamp",
        timestamp_scale: float = 1.0,
        timestamp_tolerance: float = 0.0,
        camera_key: Optional[str] = None,
    ):
        if num_frames != 9:
            raise ValueError(f"Depth requires current + 8 future frames, expected num_frames=9, got {num_frames}")
        self.dataset_root = Path(dataset_root)
        self.metadata_roots = [
            path for path in (self.dataset_root / "_meta", self.dataset_root / "metadata") if path.is_dir()
        ]
        self.images_roots = [
            path for path in (self.dataset_root / "images", self.dataset_root) if path.is_dir()
        ]
        self.image_size = tuple(int(value) for value in image_size)
        self.num_frames = int(num_frames)
        self.alignment_mode = str(alignment_mode)
        self.timestamp_scale = float(timestamp_scale)
        self.timestamp_tolerance = float(timestamp_tolerance)
        self.camera_key = None if camera_key is None else str(camera_key)
        if self.alignment_mode not in {
            "source_path",
            "token",
            "timestamp",
            "source_path_then_token_then_timestamp",
        }:
            raise ValueError(
                "`alignment_mode` must be source_path, token, timestamp, "
                "or source_path_then_token_then_timestamp"
            )
        if self.timestamp_scale <= 0:
            raise ValueError("`timestamp_scale` must be positive")
        if self.timestamp_tolerance < 0:
            raise ValueError("`timestamp_tolerance` must be non-negative")
        if not self.metadata_roots:
            raise FileNotFoundError(
                f"Expected _meta/ or metadata/ under depth dataset root: {self.dataset_root}"
            )

        records = self._load_records(metadata_glob)
        if not records:
            raise ValueError(f"No metadata records found under {self.dataset_root}")
        self._by_token: dict[str, dict[str, Any]] = {}
        self._by_source_path: dict[str, dict[str, Any]] = {}
        self._by_timestamp: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            token = self._record_token(record)
            source_path = self._record_source_path(record)
            if source_path is not None:
                for key in self._path_keys(source_path):
                    existing = self._by_source_path.get(key)
                    if existing is not None and existing is not record:
                        raise ValueError(f"Duplicate DepthNav source RGB path: {key}")
                    self._by_source_path[key] = record
            timestamp = self._record_timestamp(record)
            if token is not None:
                if token in self._by_token:
                    raise ValueError(f"Duplicate depth metadata token: {token}")
                self._by_token[token] = record
            if timestamp is not None:
                self._by_timestamp.append((timestamp, record))
        self._by_timestamp.sort(key=lambda item: item[0])
        self._timestamps = [item[0] for item in self._by_timestamp]
        if self.alignment_mode == "source_path" and not self._by_source_path:
            raise ValueError("Depth metadata has no `img_name`/`source_image_path` for source-path alignment")
        if self.alignment_mode == "token" and not self._by_token:
            raise ValueError("Depth metadata has no token/sample_id field for token alignment")
        if self.alignment_mode == "timestamp" and not self._by_timestamp:
            raise ValueError("Depth metadata has no timestamp field for timestamp alignment")

    def load(
        self,
        *,
        navsim_tokens: Iterable[str],
        navsim_timestamps: Iterable[float | int],
        navsim_image_paths: Optional[Iterable[Optional[str]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [str(token) for token in navsim_tokens]
        timestamps = [float(timestamp) for timestamp in navsim_timestamps]
        image_paths = [None] * len(tokens) if navsim_image_paths is None else list(navsim_image_paths)
        if len(tokens) != self.num_frames or len(timestamps) != self.num_frames or len(image_paths) != self.num_frames:
            raise ValueError(
                f"Expected exactly {self.num_frames} aligned depth frames, got "
                f"{len(tokens)} tokens, {len(timestamps)} timestamps, and {len(image_paths)} image paths"
            )

        frames = []
        visibility_frames = []
        for frame_index, (token, timestamp, image_path) in enumerate(zip(tokens, timestamps, image_paths)):
            record = self._match_record(
                token=token,
                timestamp=timestamp,
                source_image_path=image_path,
                frame_index=frame_index,
            )
            image, visible = self._read_image(record)
            frames.append(image)
            visibility_frames.append(visible)

        images = torch.stack(frames, dim=0)
        visible = torch.stack(visibility_frames, dim=0)
        images = self._resize_images(images)
        visible = self._resize_visibility(visible)
        depth_rgb = self._normalize_relative_depth(images, visible)
        return depth_rgb.permute(1, 0, 2, 3).contiguous(), visible.contiguous()

    def _load_records(self, metadata_glob: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for metadata_root in self.metadata_roots:
            for json_path in sorted(path for path in metadata_root.rglob(metadata_glob) if path.is_file()):
                payload = self._read_json_or_jsonl(json_path)
                for record in self._extract_records(payload):
                    if self.camera_key is not None and str(record.get("camera", record.get("camera_key", ""))) != self.camera_key:
                        continue
                    record = dict(record)
                    record["_metadata_path"] = json_path
                    records.append(record)
        return records

    @staticmethod
    def _read_json_or_jsonl(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as file:
            content = file.read().strip()
        if not content:
            return []
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return [json.loads(line) for line in content.splitlines() if line.strip()]

    @classmethod
    def _extract_records(cls, payload: Any) -> Iterable[dict[str, Any]]:
        if isinstance(payload, dict):
            if any(key in payload for key in ("image_path", "img_name", "depth_vis_name", "depth_vis_abspath")):
                yield payload
                return
            for value in payload.values():
                yield from cls._extract_records(value)
        elif isinstance(payload, list):
            for value in payload:
                yield from cls._extract_records(value)

    @staticmethod
    def _record_token(record: dict[str, Any]) -> Optional[str]:
        for key in ("navsim_token", "token", "sample_id"):
            value = record.get(key)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _record_source_path(record: dict[str, Any]) -> Optional[str]:
        for key in ("img_name", "source_image_path", "rgb_image_path"):
            value = record.get(key)
            if value is not None:
                return str(value)
        return None

    def _record_timestamp(self, record: dict[str, Any]) -> Optional[float]:
        value = record.get("timestamp")
        if value is None:
            return None
        return float(value) * self.timestamp_scale

    def _match_record(
        self,
        *,
        token: str,
        timestamp: float,
        source_image_path: Optional[str],
        frame_index: int,
    ) -> dict[str, Any]:
        record = None
        if self.alignment_mode in {"source_path", "source_path_then_token_then_timestamp"}:
            record = self._match_source_path(source_image_path)
            if record is not None:
                self._verify_timestamp(record, timestamp, token, frame_index)
                return record
            if self.alignment_mode == "source_path":
                raise KeyError(
                    f"No DepthNav metadata for NavSim RGB path={source_image_path} at sequence index={frame_index}"
                )
        if self.alignment_mode in {"token", "source_path_then_token_then_timestamp"}:
            record = self._by_token.get(token)
            if record is not None:
                self._verify_timestamp(record, timestamp, token, frame_index)
                return record
            if self.alignment_mode == "token":
                raise KeyError(f"No depth metadata for NavSim frame token={token} at sequence index={frame_index}")

        if not self._by_timestamp:
            raise KeyError(
                f"No token match for NavSim token={token}; timestamp fallback unavailable at sequence index={frame_index}"
            )
        position = bisect_left(self._timestamps, timestamp)
        candidates = []
        for index in (position - 1, position):
            if 0 <= index < len(self._by_timestamp):
                candidates.append(self._by_timestamp[index])
        matched_timestamp, record = min(candidates, key=lambda item: abs(item[0] - timestamp))
        if abs(matched_timestamp - timestamp) > self.timestamp_tolerance:
            raise KeyError(
                "Depth timestamp alignment failed at sequence index="
                f"{frame_index}: NavSim={timestamp}, nearest metadata={matched_timestamp}, "
                f"tolerance={self.timestamp_tolerance}"
            )
        return record

    def _match_source_path(self, source_image_path: Optional[str]) -> Optional[dict[str, Any]]:
        if source_image_path is None:
            return None
        for key in self._path_keys(source_image_path):
            record = self._by_source_path.get(key)
            if record is not None:
                return record
        return None

    @staticmethod
    def _path_keys(value: str) -> list[str]:
        normalized = value.replace("\\", "/").lstrip("/")
        parts = [part for part in normalized.split("/") if part]
        keys = ["/".join(parts)]
        for anchor in ("trainval", "mini", "test"):
            if anchor in parts:
                keys.append("/".join(parts[parts.index(anchor) :]))
        return list(dict.fromkeys(keys))

    def _verify_timestamp(
        self,
        record: dict[str, Any],
        expected_timestamp: float,
        token: str,
        frame_index: int,
    ) -> None:
        record_timestamp = self._record_timestamp(record)
        if record_timestamp is None:
            return
        if abs(record_timestamp - expected_timestamp) > self.timestamp_tolerance:
            raise ValueError(
                "Token matched but timestamp differs, refusing misaligned depth frame: "
                f"token={token}, sequence_index={frame_index}, NavSim={expected_timestamp}, "
                f"metadata={record_timestamp}, tolerance={self.timestamp_tolerance}"
            )

    def _read_image(self, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self._resolve_depth_image_path(record)
        array = np.asarray(Image.open(image_path))
        if array.ndim == 2:
            image = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
            visible = self._read_visibility(record, image_path, image.shape[-2:], default=image.gt(0).squeeze(0))
            return image.repeat(3, 1, 1), visible
        if array.ndim == 3 and array.shape[2] >= 3:
            image = torch.from_numpy(array[..., :3].astype(np.float32)).permute(2, 0, 1)
            visible = self._read_visibility(
                record, image_path, image.shape[-2:], default=torch.ones(image.shape[-2:], dtype=torch.bool)
            )
            return image, visible
        raise ValueError(f"Depth PNG must be HxW or HxWx3, got shape {array.shape} at {image_path}")

    def _read_visibility(
        self,
        record: dict[str, Any],
        image_path: Path,
        shape: tuple[int, int],
        *,
        default: torch.Tensor,
    ) -> torch.Tensor:
        path_value = record.get("visibility_path")
        if path_value is None:
            return default.to(dtype=torch.bool)
        visibility_path = self._resolve_image_path(str(path_value), fallback_parent=image_path.parent)
        array = np.asarray(Image.open(visibility_path), dtype=np.uint8)
        if array.ndim == 3:
            array = array[..., 0]
        if tuple(array.shape) != tuple(shape):
            raise ValueError(f"Visibility shape mismatch at {visibility_path}: {array.shape} vs {shape}")
        return torch.from_numpy(array > 0)

    def _resolve_image_path(self, value: str, fallback_parent: Optional[Path] = None) -> Path:
        path = Path(value)
        candidates = [path] if path.is_absolute() else [
            self.dataset_root / path,
            *(image_root / path for image_root in self.images_roots),
        ]
        if fallback_parent is not None and not path.is_absolute():
            candidates.append(fallback_parent / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Metadata image_path does not exist: {value}; searched {candidates}")

    def _resolve_depth_image_path(self, record: dict[str, Any]) -> Path:
        for value in (record.get("depth_vis_abspath"), record.get("depth_vis_name"), record.get("image_path")):
            if value is None:
                continue
            path = Path(str(value))
            if path.is_file():
                return path
            for image_root in self.images_roots:
                candidate = image_root / path.name
                if candidate.is_file():
                    return candidate
                if not path.is_absolute() and (image_root / path).is_file():
                    return image_root / path
        raise FileNotFoundError(
            "No usable depth image in metadata record; expected depth_vis_abspath, depth_vis_name, or image_path"
        )

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        return F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)

    def _resize_visibility(self, visibility: torch.Tensor) -> torch.Tensor:
        return F.interpolate(visibility.unsqueeze(1).float(), size=self.image_size, mode="nearest").squeeze(1).gt(0.5)

    @staticmethod
    def _normalize_relative_depth(images: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 3 and float(images.max()) <= 255.0:
            return images.div(127.5).sub(1.0).masked_fill(~visible.unsqueeze(1), -1.0)
        valid = images[:, :1][visible.unsqueeze(1).expand(-1, 1, -1, -1)]
        if valid.numel() == 0:
            return torch.full_like(images, -1.0)
        low = torch.quantile(valid, 0.01)
        high = torch.quantile(valid, 0.99)
        normalized = images.sub(low).div((high - low).clamp_min(1e-6)).mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        return normalized.masked_fill(~visible.unsqueeze(1), -1.0)
