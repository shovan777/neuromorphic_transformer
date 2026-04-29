import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
from config import DEVICE, SEED, MODEL_CONFIG, INPUT_CHANNELS, IMG_SIZE, USE_POISSON_ENCODING
from config import set_seed
from dataset import get_cifar10_dataloaders
from model import SpikFormer
from spikingjelly.activation_based import functional

CHECKPOINT = (
    "models/checkpoint_20260427-01h/"
    "best_spikformer_dim128_heads4_mlp256_layers4_epoch198_acc0.8551.pth"
)
NUM_IMAGES = 5       # rows in the figure
OUTPUT_DIR = "visualizations_new"
OUTPUT_FILE = "attention_map_grid.png"


def get_spatial_attention(attn_weights):
    """
    attn_weights: [T, B, num_heads, N, N]
    Returns per-image spatial attention map upsampled to IMG_SIZE x IMG_SIZE.
    Shape: [B, IMG_SIZE, IMG_SIZE]
    """
    # Average over time steps and heads -> [B, N, N]
    attn = attn_weights.mean(dim=0).mean(dim=1)
    # Mean over query positions -> [B, N] (how much each patch is attended to)
    attn = attn.mean(dim=1)
    patch_grid = IMG_SIZE // MODEL_CONFIG["patch_size"]   # 8 for CIFAR-10
    attn = attn.reshape(-1, 1, patch_grid, patch_grid)   # [B, 1, 8, 8]
    attn = F.interpolate(attn, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return attn.squeeze(1)   # [B, H, W]


def denormalize(img_tensor):
    """Undo CIFAR-10 normalization for display."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_attention():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use the first 5 train images (indices 0-4)
    train_loader, test_loader, num_classes, class_names = get_cifar10_dataloaders(
        normalize=not USE_POISSON_ENCODING
    )
    images, labels = [], []
    for data, target in test_loader:
        for img, lbl in zip(data, target):
            images.append(img)
            labels.append(lbl.item())
            if len(images) == NUM_IMAGES:
                break
        if len(images) == NUM_IMAGES:
            break

    mlp_ratio = MODEL_CONFIG["mlp_dim"] / MODEL_CONFIG["embed_dim"]
    model = SpikFormer(
        num_classes=num_classes,
        in_channels=INPUT_CHANNELS,
        num_channels=MODEL_CONFIG["embed_dim"],
        img_size=IMG_SIZE,
        patch_size=MODEL_CONFIG["patch_size"],
        dim=MODEL_CONFIG["embed_dim"],
        depth=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        mlp_ratio=mlp_ratio,
        use_poisson=USE_POISSON_ENCODING,
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    data = torch.stack(images).to(DEVICE)

    functional.reset_net(model)
    with torch.no_grad():
        output = model(data)
    preds = output.argmax(dim=1)

    attn_weights = model.blocks[-1].attn.attn_weights   # [T, B, heads, N, N]
    spatial_attn = get_spatial_attention(attn_weights).cpu().numpy()   # [B, H, W]

    # Layout: NUM_IMAGES rows x 2 columns
    fig = plt.figure(figsize=(3.2, NUM_IMAGES * 1.6))
    gs = gridspec.GridSpec(NUM_IMAGES, 2, figure=fig,
                           hspace=0.05, wspace=0.04,
                           left=0.01, right=0.99, top=0.93, bottom=0.01)

    fig.text(0.25, 0.965, "Input\nImage",   ha="center", va="center",
             fontsize=8, fontweight="bold")
    fig.text(0.75, 0.965, "Attention\nMap", ha="center", va="center",
             fontsize=8, fontweight="bold")

    for i in range(NUM_IMAGES):
        ax_img  = fig.add_subplot(gs[i, 0])
        ax_attn = fig.add_subplot(gs[i, 1])

        img_np = denormalize(images[i])   # [H, W, 3] in [0,1]

        ax_img.imshow(img_np)
        correct = u"\u2713" if preds[i].item() == labels[i] else u"\u2717"
        ax_img.set_ylabel(f"{class_names[labels[i]]} {correct}",
                          fontsize=6, rotation=0, labelpad=36, va="center")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)

        # Normalise attention to [0, 1] and use as brightness mask over the image
        attn_map = spatial_attn[i]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        # Multiply each colour channel by the attention mask: bright where attended
        overlay = img_np * attn_map[:, :, np.newaxis]
        overlay = np.clip(overlay, 0, 1)

        ax_attn.imshow(overlay)
        ax_attn.set_xticks([])
        ax_attn.set_yticks([])
        for spine in ax_attn.spines.values():
            spine.set_visible(False)

    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    visualize_attention()
