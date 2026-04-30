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
from spikingjelly.activation_based import functional, neuron
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
_spike_rate_cache = {}

def _register_hooks(m):
    handles = []
    # Capture output of the 3-stage spike conv stem (Img2Spike)
    handles.append(m.img2spike.register_forward_hook(
        lambda mod, inp, out: _spike_enc_cache.update({"img2spike": out.detach().cpu()})
    ))

    # Capture spike rates per transformer block from all spiking nodes in that block.
    _spike_rate_cache.clear()
    _spike_rate_cache["block_fired"] = [0.0 for _ in range(len(m.blocks))]
    _spike_rate_cache["block_total"] = [0 for _ in range(len(m.blocks))]

    def _make_block_hook(block_idx):
        def hook(_, __, output):
            spikes = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(spikes):
                return
            spikes = spikes.detach()
            _spike_rate_cache["block_fired"][block_idx] += spikes.sum().item()
            _spike_rate_cache["block_total"][block_idx] += spikes.numel()
        return hook

    for block_idx, block in enumerate(m.blocks):
        for module in block.modules():
            if isinstance(module, neuron.BaseNode):
                handles.append(module.register_forward_hook(_make_block_hook(block_idx)))

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


def _plot_spike_encoding(spike_tensor, image_np, class_name, label):
    """
    Match visualize_cifar10_spikes.py format:
      - first panel: original image
      - then one panel per timestep for the most active channel
    """
    # spike_tensor: [T, B, C, H, W]
    channel_totals = spike_tensor[:, 0].sum(dim=(0, 2, 3))
    active_channel = int(channel_totals.argmax().item())
    spike_maps = spike_tensor[:, 0, active_channel].numpy()  # [T, H, W]

    num_timesteps = spike_maps.shape[0]
    cols = 3
    rows = (num_timesteps + 2) // cols  # +1 for original image
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=120)
    axes = np.atleast_1d(axes).flatten()

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


def _plot_firing_heatmap(model, block_rates, overall_rate):
    """
    Plot attention activity intensity with explicit axes:
      - y-axis: transformer blocks (0..NUM_LAYERS-1)
      - x-axis: patch indices (0..N-1)
    """
    all_rates = []
    for block in model.blocks:
        attn = block.attn.attn_weights  # [T, B, heads, N, N]
        # Mean activity per patch (sum over key dim, avg over T and heads)
        rate = attn[:, 0].mean(dim=0).sum(dim=-1)  # [heads, N]
        rate = rate.mean(dim=0).cpu().numpy()  # [N]
        all_rates.append(rate)

    arr = np.stack(all_rates, axis=0)  # [L, N]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=120, gridspec_kw={"width_ratios": [2.2, 1.0]})

    # Left: attention activity heatmap (patch x block)
    ax = axes[0]
    im = ax.imshow(arr, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title("Attention Activity per Patch per Block")
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Block")
    ax.set_yticks(list(range(arr.shape[0])))
    ax.set_yticklabels([f"Block {i}" for i in range(arr.shape[0])])
    xticks = np.linspace(0, arr.shape[1] - 1, num=9, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    fig.colorbar(im, ax=ax, label="magnitude", fraction=0.03, pad=0.02)

    # Right: true spike firing rates per block + overall average
    ax2 = axes[1]
    y = np.arange(len(block_rates))
    bars = ax2.barh(y, np.array(block_rates) * 100.0, color="#4caf50")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"Block {i}" for i in y])
    ax2.set_xlabel("Firing rate (%)")
    ax2.set_title("Spike Firing Rate")
    ax2.axvline(overall_rate * 100.0, color="red", linestyle="--", linewidth=1.5, label=f"Overall {overall_rate*100:.2f}%")
    ax2.legend(loc="lower right", fontsize=8)
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8)

    fig.tight_layout()
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

    block_fired = _spike_rate_cache.get("block_fired", [])
    block_total = _spike_rate_cache.get("block_total", [])
    block_rates = [
        (f / t) if t > 0 else 0.0
        for f, t in zip(block_fired, block_total)
    ]
    overall_rate = (
        sum(block_fired) / sum(block_total)
        if sum(block_total) > 0
        else 0.0
    )

    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_name = CIFAR10_CLASSES[pred_idx]
    true_name = CIFAR10_CLASSES[true_label]
    conf = float(probs[pred_idx]) * 100

    label_text = (
        f"**Prediction:** {pred_name} ({conf:.1f}%)\n"
        f"**Ground truth:** {true_name}  "
        + ("✅" if pred_idx == true_label else "❌")
        + "\n"
        + f"**Overall block firing rate:** {overall_rate*100:.3f}%"
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
    enc_img = _plot_spike_encoding(
        _spike_enc_cache["img2spike"],
        img_np,
        true_name,
        true_label,
    )

    # Attention overlay (no_grad already done, attn_weights stored in model)
    attn_img = _plot_attention_overlay(img_np, model)

    # Firing heatmap
    fire_img = _plot_firing_heatmap(model, block_rates=block_rates, overall_rate=overall_rate)

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
        fire_out  = gr.Image(label="Attention Activity + Spike Firing Rates")

    btn.click(
        fn=run_inference,
        inputs=[btn],
        outputs=[img_out, label_out, pred_out, enc_out, attn_out, fire_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
