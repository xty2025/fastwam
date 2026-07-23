"""CPU/GPU smoke run for the external-depth data contract.

It intentionally avoids loading the 5B checkpoint while exercising the exact
separate-root depth layout, current-plus-eight-future alignment, visibility
masking, VAE-compatible three-channel tensor contract, and an optimizer epoch.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fastwam.datasets.depth_sequence import MetadataDepthSequence


class MockDepthDataset(Dataset):
    def __init__(self, root: str, image_size: tuple[int, int]):
        self.loader = MetadataDepthSequence(root, image_size=image_size)
        self.sample_ids = sorted(
            path.name.split("-000000depth.jpg")[0]
            for path in Path(root).glob("*-000000depth.jpg")
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        depth_rgb, depth_visible = self.loader.load(
            navsim_tokens=[f"unused-{frame_index}" for frame_index in range(9)],
            navsim_timestamps=[index * 10_000_000 + frame_index * 500_000 for frame_index in range(9)],
            navsim_image_paths=[
                f"s/trainval/mock_scene/CAM_F0/{sample_id}-{frame_index:06d}.jpg"
                for frame_index in range(9)
            ],
        )
        return {"depth_rgb": depth_rgb, "depth_visible": depth_visible}


class TinyDepthDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, noisy_depth):
        batch_size, channels, num_frames, height, width = noisy_depth.shape
        frames = noisy_depth.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        prediction = self.net(frames)
        return prediction.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = MockDepthDataset(args.depth_root, image_size=(64, 96))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = TinyDepthDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        print(f"epoch={epoch + 1}/{args.epochs} depth_frames=current+8_future")
        for batch in loader:
            depth = batch["depth_rgb"].to(device)
            visible = batch["depth_visible"].to(device).unsqueeze(1).expand_as(depth)
            noise = torch.randn_like(depth)
            noisy_depth = depth + 0.1 * noise
            prediction = model(noisy_depth)
            loss = ((prediction - noise).square() * visible).sum() / visible.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            print(f"step_loss={loss.item():.6f}")
    print("DEPTH_SMOKE_EPOCH_COMPLETED")


if __name__ == "__main__":
    main()
