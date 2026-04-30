import sys
import os
sys.path.insert(0, "/home/srs476/vit_exp/vit_archives/vit_nas_before_may_21")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modules.super_net import SuperNet

CHECKPOINT = (
    "/home/srs476/vit_exp/vit_archives/vit_nas_before_may_21/"
    "final_supernet_embed512_heads8_mlp1024_layers6_epoch200_acc0.91_20260420-19h.pth"
)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 32
PATCH_SIZE = 4
NUM_IMAGES = 5
OUTPUT_DIR = "visualizations_new"
OUTPUT_FILE = "attention_map_ann.png"

# Normalization used during ANN training
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_train_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    return DataLoader(dataset, batch_size=NUM_IMAGES, shuffle=False)


def denormalize(img_tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def get_spatial_attention(attn_weights):
    """
    attn_weights: [B, num_heads, N+1, N+1]  (N+1 because of CLS token)
    Returns upsampled spatial attention [B, IMG_SIZE, IMG_SIZE].
    """
    # Drop CLS token row and column -> [B, num_heads, N, N]
    attn = attn_weights[:, :, 1:, 1:]
    # Average over heads -> [B, N, N]
    attn = attn.mean(dim=1)
    # Average over query positions -> [B, N]
    attn = attn.mean(dim=1)
    patch_grid = IMG_SIZE // PATCH_SIZE   # 8
    attn = attn.reshape(-1, 1, patch_grid, patch_grid)
    attn = F.interpolate(attn, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return attn.squeeze(1)   # [B, H, W]


def visualize_attention():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model = SuperNet(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        mlp_dim=1024,
        num_classes=10,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # Hook to capture attention from the last transformer block
    attn_cache = {}

    def attn_hook(module, input, output):
        # Re-run the attention computation to grab the weights
        x = input[0]
        B, T, _ = x.shape
        qkv = (
            module.qkv_linear(x)
            .reshape(B, T, 3, module.active_num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, _ = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * (module.head_dim ** -0.5)
        attn_cache["weights"] = F.softmax(attn, dim=-1).detach()  # [B, heads, N+1, N+1]

    hook = model.transformer_blocks[-1].mha.register_forward_hook(attn_hook)

    # Get first 5 training images
    loader = get_train_loader()
    data, target = next(iter(loader))
    data, target = data[:NUM_IMAGES].to(DEVICE), target[:NUM_IMAGES]

    with torch.no_grad():
        output = model(data)
    preds = output.argmax(dim=1).cpu()
    hook.remove()

    spatial_attn = get_spatial_attention(attn_cache["weights"]).cpu().numpy()

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

        img_np = denormalize(data[i])

        ax_img.imshow(img_np)
        correct = "\u2713" if preds[i].item() == target[i].item() else "\u2717"
        ax_img.set_ylabel(f"{CIFAR10_CLASSES[target[i].item()]} {correct}",
                          fontsize=6, rotation=0, labelpad=36, va="center")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)

        # Attention overlay: bright = high attention, dark = low attention
        attn_map = spatial_attn[i]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        overlay = np.clip(img_np * attn_map[:, :, np.newaxis], 0, 1)

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
