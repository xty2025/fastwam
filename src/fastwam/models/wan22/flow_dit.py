from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import DiTBlock, precompute_freqs_cis, sinusoidal_embedding_1d


class FlowDiT(nn.Module):
    """Patch DiT expert for BEV motion-flow denoising inside MoT."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        text_dim: int = 4096,
        freq_dim: int = 256,
        num_layers: int = 30,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        patch_size: int = 4,
        bev_size: tuple[int, int] | list[int] = (200, 200),
        eps: float = 1.0e-6,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.patch_size = int(patch_size)
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if self.attn_head_dim <= 0 or self.attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be positive and even, got {self.attn_head_dim}")
        if self.bev_size[0] % self.patch_size != 0 or self.bev_size[1] % self.patch_size != 0:
            raise ValueError(
                f"`bev_size` must be divisible by patch_size, got {self.bev_size} and {self.patch_size}"
            )

        patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        grid_h = self.bev_size[0] // self.patch_size
        grid_w = self.bev_size[1] // self.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_h * grid_w, self.hidden_dim))
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=float(eps),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.out = nn.Linear(self.hidden_dim, patch_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=grid_h * grid_w + 1024)

    def pre_dit(
        self,
        flow: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if flow.ndim != 4:
            raise ValueError(f"`flow` must be [B, C, H, W], got {tuple(flow.shape)}")
        if flow.shape[1] != self.in_channels:
            raise ValueError(f"`flow` channel dim must be {self.in_channels}, got {flow.shape[1]}")
        if tuple(flow.shape[-2:]) != self.bev_size:
            raise ValueError(f"`flow` spatial dims must be {self.bev_size}, got {tuple(flow.shape[-2:])}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B, L, D], got {tuple(context.shape)}")

        batch_size = flow.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(f"Batch mismatch between flow and context: {batch_size} vs {context.shape[0]}")
        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        elif context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`context_mask` must be [B, L] matching context, got {tuple(context_mask.shape)} "
                f"vs {tuple(context.shape)}"
            )

        tokens = self.patch_embed(flow).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        freqs = self.freqs[: tokens.shape[1]].view(tokens.shape[1], 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "grid_size": (self.bev_size[0] // self.patch_size, self.bev_size[1] // self.patch_size),
                "seq_len": tokens.shape[1],
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        patches = self.out(self.norm(tokens))
        return self._unpatchify(patches)

    def forward(
        self,
        flow: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            flow=flow,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    context_mask=pre_state["context_mask"],
                )
            else:
                x = block(
                    x,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    context_mask=pre_state["context_mask"],
                )
        return self.post_dit(x, pre_state)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b, _, _ = patches.shape
        p = self.patch_size
        h = self.bev_size[0] // p
        w = self.bev_size[1] // p
        x = patches.view(b, h, w, self.in_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(b, self.in_channels, self.bev_size[0], self.bev_size[1])
