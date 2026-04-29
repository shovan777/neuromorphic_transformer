import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from config import (
    DATA_DIR,
    DEVICE,
    INPUT_CHANNELS,
    NUM_CHANNELS,
    SEED,
    set_seed,
    USE_POISSON_ENCODING,
)
from model import Img2Spike
from spikingjelly.activation_based import functional


def main():
    set_seed(SEED)

    # raw images for display
    test_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    # normalized images for spike generation (matches training pipeline)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    norm_ops = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
    if not USE_POISSON_ENCODING:
        norm_ops.append(transforms.Normalize(mean=mean, std=std))
    norm_test_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=transforms.Compose(norm_ops),
        download=True,
    )

    img2spike = Img2Spike(
        in_channels=INPUT_CHANNELS,
        channels=NUM_CHANNELS,
        use_poisson=USE_POISSON_ENCODING,
    ).to(DEVICE)
    img2spike.train()

    output_dir = "visualizations_new"
    os.makedirs(output_dir, exist_ok=True)

    for sample_idx in range(10):
        image, label = test_set[sample_idx]
        norm_image, _ = norm_test_set[sample_idx]
        image_batch = norm_image.unsqueeze(0).to(DEVICE)

        functional.reset_net(img2spike)
        with torch.no_grad():
            spikes = img2spike(image_batch)  # [T, B, C, H, W]

        channel_totals = spikes[:, 0].sum(dim=(0, 2, 3))
        active_channel = int(channel_totals.argmax().item())
        spike_maps = spikes[:, 0, active_channel].detach().cpu().numpy()  # [T, H, W]
        image_np = image.permute(1, 2, 0).numpy()
        class_name = test_set.classes[label]
        mode = "poisson" if USE_POISSON_ENCODING else "mslif"
        output_path = os.path.join(
            output_dir, f"cifar10_img2spike_{mode}_sample_{sample_idx:02d}_{class_name}.png"
        )

        num_timesteps = spike_maps.shape[0]
        cols = 3
        rows = (num_timesteps + 2) // cols  # +1 for original image
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()

        axes[0].imshow(image_np)
        axes[0].set_title(f"Original CIFAR10 (label={label}: {class_name})")
        axes[0].axis("off")

        for t in range(num_timesteps):
            ax = axes[t + 1]
            ax.imshow(spike_maps[t], cmap="gray")
            ax.set_title(f"Spike map S[{t}] (ch={active_channel})")
            ax.axis("off")

        for k in range(num_timesteps + 1, len(axes)):
            axes[k].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        fired = spikes.sum().item()
        total = spikes.numel()
        print(
            f"Saved CIFAR10 spike visualization to: {output_path} "
            f"| spike_rate={fired / total:.6f} | active_ch={active_channel}"
        )


if __name__ == "__main__":
    main()
