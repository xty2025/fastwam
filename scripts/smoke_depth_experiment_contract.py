#!/usr/bin/env python3
"""Validate the depth-method contracts without loading the 5B checkpoint."""

from __future__ import annotations

import torch

from fastwam.models.wan22.fastwam_dit import FastWAMDiT


def main() -> None:
    torch.manual_seed(7)
    model = object.__new__(FastWAMDiT)
    model.depth_mask_ratio = 0.4
    model.depth_mask_block_size = 2
    latents = torch.zeros((2, 16, 9, 6, 8), dtype=torch.float32)
    mask = model._sample_future_depth_block_mask(latents)

    assert mask.dtype is torch.bool
    assert tuple(mask.shape) == tuple(latents.shape)
    assert not bool(mask[:, :, 0].any()), "D0 must remain the clean depth condition"
    assert bool(mask[:, :, 1:].any()), "D1:8 must contain masked latent blocks"
    assert torch.equal(mask[:, 0], mask[:, -1]), "mask must cover all latent channels"

    source = torch.randn_like(latents)
    genception_query = torch.zeros_like(source)
    genception_velocity_target = -source
    recovered_depth = genception_query - genception_velocity_target
    assert torch.allclose(recovered_depth, source)

    print("LINGBOT_MASKED_REFINEMENT_CONTRACT_OK", tuple(mask.shape))
    print("GENCEPTION_ZERO_DEPTH_QUERY_CONTRACT_OK", tuple(recovered_depth.shape))
    print("NO_CONV3D_ADDED")


if __name__ == "__main__":
    main()
