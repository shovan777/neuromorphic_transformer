"""
SpikFormer CIFAR-10 Demo
------------------------
Click "Random Image" to:
  - Pick a random CIFAR-10 test image
  - Run it through the trained SpikFormer
  - Display: original image, spike encoding, attention overlay, neuron firing heatmap
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from spikingjelly.activation_based import functional
from spikingjelly import visualizing
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Config (mirrors config.py values used for the best checkpoint)
# ---------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 32
PATCH_SIZE  = 4
DIM         = 128
NUM_HEADS   = 4
MLP_DIM     = 256
NUM_LAYERS  = 4
TIME_STEP   = 4

CHECKPOINT  = (
    "models/checkpoint_20260427-01h/"
    "best_spikformer_dim128_heads4_mlp256_layers4_epoch198_acc0.8551.pth"
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from model import SpikFormer

mlp_ratio = MLP_DIM / DIM
model = SpikFormer(
    num_classes=10, in_channels=3, num_channels=DIM,
    img_size=IMG_SIZE, patch_size=PATCH_SIZE,
    dim=DIM, depth=NUM_LAYERS, num_heads=NUM_HEADS,
    mlp_ratio=mlp_ratio,
).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print(f"Model loaded from {CHECKPOINT}")

# ---------------------------------------------------------------------------
# Load CIFAR-10 test set (no augmentation, no normalisation — we'll normalise
# manually so we can also display the raw pixel image)
# ---------------------------------------------------------------------------
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=False,
    transform=transforms.ToTensor(),  # [0,1] only — we normalise below
)


# ---------------------------------------------------------------------------
# Hook to capture spike feature maps from Img2Spike output
# ---------------------------------------------------------------------------
_spike_enc_cache = {}

def _register_hooks(m):
    handles = []
    # Capture output of the 3-stage spike conv stem (Img2Spike)
    handles.append(m.img2spike.register_forward_hook(
        lambda mod, inp, out: _spike_enc_cache.update({"img2spike": out.detach().cpu()})
    ))
    return handles


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fig_to_numpy(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return arr[:, :, :3]  # drop alpha


def _plot_spike_encoding(spike_tensor):
    """
    spike_tensor: [T, B, C, H, W]  — take B=0, average over C → [T, H, W]
    Use spikingjelly plot_2d_feature_map to show T frames side-by-side.
    """
    T, _, C, H, W = spike_tensor.shape
    # Average over channels to get [T, H, W], then layout as [1, T] grid
    frames = spike_tensor[:, 0].mean(dim=1).numpy()  # [T, H, W]
    # Stack into a [T*H, W] image for plot_2d_feature_map (nrows=1, ncols=T)
    grid = np.concatenate([frames[t] for t in range(T)], axis=1)  # [H, T*W]
    fig, ax = plt.subplots(1, 1, figsize=(10, 2.5), dpi=120)
    im = ax.imshow(grid, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"Spike Encoding  (T={T} timesteps, avg over {C} channels)", fontsize=10)
    ax.set_xticks([W * t + W // 2 for t in range(T)])
    ax.set_xticklabels([f"t={t}" for t in range(T)])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    out = _fig_to_numpy(fig)
    plt.close(fig)
    return out


def _plot_attention_overlay(img_np, model):
    """
    img_np: HxWxC uint8 (original image, no normalisation)
    Averages attention weights over T and heads, upsamples to image size.
    """
    N = (IMG_SIZE // PATCH_SIZE) ** 2  # 64 patches
    grid_h = grid_w = IMG_SIZE // PATCH_SIZE  # 8x8

    fig, axes = plt.subplots(1, NUM_LAYERS + 1, figsize=(3 * (NUM_LAYERS + 1), 3), dpi=120)
    axes[0].imshow(img_np)
    axes[0].set_title("Input", fontsize=9)
    axes[0].axis("off")

    for i, block in enumerate(model.blocks):
        attn = block.attn.attn_weights  # [T, B, heads, N, N]  (B=1)
        attn_map = attn[:, 0].mean(dim=0).mean(dim=0)  # avg T, avg heads → [N, N]
        # Row-mean: how much each patch attends on average
        patch_scores = attn_map.mean(dim=-1).cpu().numpy()  # [N]
        patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)
        heat = patch_scores.reshape(grid_h, grid_w)
        heat_up = F.interpolate(
            torch.tensor(heat).unsqueeze(0).unsqueeze(0).float(),
            size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        overlay = (img_np / 255.0) * heat_up[:, :, None]
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"Block {i}", fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle("Attention Overlay per Block", fontsize=10, y=1.02)
    plt.tight_layout()
    out = _fig_to_numpy(fig)
    plt.close(fig)
    return out


def _plot_firing_heatmap(model):
    """
    For each transformer block, collect mean firing rate per neuron position (N)
    from q/k/v LIF outputs and plot as a heatmap using spikingjelly.
    Uses the stored attn_weights as a proxy signal.
    """
    all_rates = []
    for block in model.blocks:
        attn = block.attn.attn_weights  # [T, B, heads, N, N]
        # Mean activation per patch (sum over key dim, avg over T, heads, B)
        rate = attn[:, 0].mean(dim=0).sum(dim=-1)  # [heads, N]
        rate = rate.mean(dim=0).cpu().numpy()       # [N]
        all_rates.append(rate)

    arr = np.stack(all_rates, axis=0)  # [L, N]
    fig = visualizing.plot_2d_heatmap(
        array=arr,
        title="Mean Attention Activity per Patch per Block",
        xlabel="Patch index",
        ylabel="Block",
        int_x_ticks=False,
        int_y_ticks=True,
        figsize=(10, 3),
        dpi=120,
    )
    out = _fig_to_numpy(fig)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main inference + visualisation function
# ---------------------------------------------------------------------------

def run_inference(_):
    """Called on button click. Returns all output panel images + label."""
    idx = np.random.randint(len(test_dataset))
    img_tensor, true_label = test_dataset[idx]  # [3, H, W] in [0,1]

    # Normalise for model input
    norm = transforms.Normalize(MEAN, STD)
    x = norm(img_tensor).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    # Raw image for display (uint8)
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Register hook to capture spike encoding
    _spike_enc_cache.clear()
    hooks = _register_hooks(model)

    with torch.no_grad():
        functional.reset_net(model)
        logits = model(x)  # hooks fire during this call

    for h in hooks:
        h.remove()

    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_name = CIFAR10_CLASSES[pred_idx]
    true_name = CIFAR10_CLASSES[true_label]
    conf = float(probs[pred_idx]) * 100

    label_text = (
        f"**Prediction:** {pred_name} ({conf:.1f}%)\n"
        f"**Ground truth:** {true_name}  "
        + ("✅" if pred_idx == true_label else "❌")
    )

    # Top-3 bar chart
    top3_idx = probs.argsort()[::-1][:3]
    fig_pred, ax = plt.subplots(figsize=(4, 2.5), dpi=120)
    colors = ["#2196F3" if i == pred_idx else "#90CAF9" for i in top3_idx]
    bars = ax.barh([CIFAR10_CLASSES[i] for i in top3_idx], probs[top3_idx], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Top-3 Predictions")
    for bar, p in zip(bars, probs[top3_idx]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    pred_img = _fig_to_numpy(fig_pred)
    plt.close(fig_pred)

    # Spike encoding
    enc_img = _plot_spike_encoding(_spike_enc_cache["img2spike"])

    # Attention overlay (no_grad already done, attn_weights stored in model)
    attn_img = _plot_attention_overlay(img_np, model)

    # Firing heatmap
    fire_img = _plot_firing_heatmap(model)

    return img_np, label_text, pred_img, enc_img, attn_img, fire_img


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="SpikFormer CIFAR-10 Demo") as demo:
    gr.Markdown("## 🧠 SpikFormer CIFAR-10 Demo\nSpiking Vision Transformer · 85.51% test accuracy")

    with gr.Row():
        btn = gr.Button("🎲 Random Image", variant="primary", scale=0)
        label_out = gr.Markdown("Click the button to start.")

    with gr.Row():
        img_out   = gr.Image(label="Input Image", width=160, height=160)
        pred_out  = gr.Image(label="Top-3 Predictions")

    with gr.Row():
        enc_out   = gr.Image(label="Spike Encoding (Img2Spike output, T=4)")

    with gr.Row():
        attn_out  = gr.Image(label="Attention Overlay per Block")

    with gr.Row():
        fire_out  = gr.Image(label="Attention Firing Heatmap (patch × block)")

    btn.click(
        fn=run_inference,
        inputs=[btn],
        outputs=[img_out, label_out, pred_out, enc_out, attn_out, fire_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
