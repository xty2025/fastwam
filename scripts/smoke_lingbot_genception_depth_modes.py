#!/usr/bin/env python3
"""One-step I/O-contract smoke tests for LingBot-style and GenCeption-style depth modes.

This is deliberately a lightweight mechanism check, not a replacement for a
pretrained FastWAM training run:
  - LingBot-style: RGB + masked relative-depth RGB -> reconstructed depth RGB.
  - GenCeption-style: RGB video only -> depth-RGB video.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBToDepthProbe(nn.Module):
    """GenCeption-style RGB-in/depth-out probe, applied independently per frame."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, rgb_video: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = rgb_video.shape
        flat = rgb_video.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        output = self.net(flat)
        return output.reshape(batch_size, frames, 3, height, width).permute(0, 2, 1, 3, 4)


class MaskedDepthRefinerProbe(nn.Module):
    """LingBot-style RGB + masked-depth -> depth probe, applied per frame."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(
        self,
        rgb_video: torch.Tensor,
        masked_depth_video: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, frames, height, width = rgb_video.shape
        flat = torch.cat((rgb_video, masked_depth_video, mask), dim=1)
        flat = flat.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, 7, height, width)
        output = self.net(flat)
        return output.reshape(batch_size, frames, 3, height, width).permute(0, 2, 1, 3, 4)


def block_mask(batch_size: int, frames: int, height: int, width: int, ratio: float, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, 1, frames, height, width), device=device)
    block_height = max(height // 4, 1)
    block_width = max(width // 4, 1)
    blocks_per_frame = max(int((height * width * ratio) / (block_height * block_width)), 1)
    for batch_index in range(batch_size):
        for frame_index in range(frames):
            for _ in range(blocks_per_frame):
                top = torch.randint(0, max(height - block_height + 1, 1), ()).item()
                left = torch.randint(0, max(width - block_width + 1, 1), ()).item()
                mask[batch_index, :, frame_index, top : top + block_height, left : left + block_width] = 1.0
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device(args.device)
    rgb_video = torch.randn(args.batch_size, 3, args.frames, args.height, args.width, device=device)
    depth_target = torch.randn_like(rgb_video)
    depth_mask = block_mask(
        args.batch_size, args.frames, args.height, args.width, args.mask_ratio, device
    )
    masked_depth = depth_target * (1.0 - depth_mask)

    genception = RGBToDepthProbe().to(device)
    lingbot = MaskedDepthRefinerProbe().to(device)
    optimizer = torch.optim.AdamW(
        list(genception.parameters()) + list(lingbot.parameters()),
        lr=1e-3,
    )

    predicted_from_rgb = genception(rgb_video)
    predicted_refined_depth = lingbot(rgb_video, masked_depth, depth_mask)
    loss_rgb_to_depth = F.mse_loss(predicted_from_rgb, depth_target)
    masked_error = (predicted_refined_depth - depth_target).square() * depth_mask
    loss_masked_refinement = masked_error.sum() / depth_mask.expand_as(masked_error).sum().clamp_min(1.0)
    total_loss = loss_rgb_to_depth + loss_masked_refinement
    total_loss.backward()
    optimizer.step()

    print("GENCEPTION_RGB_IN_DEPTH_OUT_OK", tuple(rgb_video.shape), tuple(predicted_from_rgb.shape))
    print("LINGBOT_RGB_MASKED_DEPTH_IN_DEPTH_OUT_OK", tuple(masked_depth.shape), tuple(predicted_refined_depth.shape))
    print(f"loss_rgb_to_depth={loss_rgb_to_depth.item():.6f}")
    print(f"loss_masked_refinement={loss_masked_refinement.item():.6f}")
    print("DEPTH_IO_MODE_SMOKE_COMPLETED")


if __name__ == "__main__":
    main()
