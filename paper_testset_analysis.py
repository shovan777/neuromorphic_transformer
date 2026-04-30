import argparse
import json
import os
import types
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import model as model_def
from config import (
    DEVICE,
    IMG_SIZE,
    INPUT_CHANNELS,
    MODEL_CONFIG,
    NUM_WORKERS,
    USE_POISSON_ENCODING,
)
from model import SpikFormer


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class EvalResult:
    accuracy: float
    preds: np.ndarray
    labels: np.ndarray
    probs: np.ndarray
    confidences: np.ndarray


def get_test_loader(batch_size: int) -> DataLoader:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if not USE_POISSON_ENCODING:
        ops.append(transforms.Normalize(mean=mean, std=std))
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(ops),
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


def build_model(checkpoint: str) -> SpikFormer:
    mlp_ratio = MODEL_CONFIG["mlp_dim"] / MODEL_CONFIG["embed_dim"]
    model = SpikFormer(
        num_classes=10,
        in_channels=INPUT_CHANNELS,
        num_channels=MODEL_CONFIG["embed_dim"],
        img_size=IMG_SIZE,
        patch_size=MODEL_CONFIG["patch_size"],
        dim=MODEL_CONFIG["embed_dim"],
        depth=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        mlp_ratio=mlp_ratio,
        use_poisson=USE_POISSON_ENCODING,
        use_cupy=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model


def evaluate(model: SpikFormer, loader: DataLoader) -> EvalResult:
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            functional.reset_net(model)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    confidences = probs[np.arange(len(preds)), preds]
    accuracy = float((preds == labels).mean())
    return EvalResult(accuracy=accuracy, preds=preds, labels=labels, probs=probs, confidences=confidences)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion(cm: np.ndarray, out_path: str, normalize: bool = False):
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        mat = cm / row_sum
        title = "Confusion Matrix (row-normalized)"
        cbar_label = "fraction"
        fmt = "{:.2f}"
    else:
        mat = cm
        title = "Confusion Matrix (counts)"
        cbar_label = "count"
        fmt = "{:d}"

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=200)
    im = ax.imshow(mat, cmap="Blues", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = fmt.format(mat[i, j]) if normalize else fmt.format(int(mat[i, j]))
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_class_metrics(cm: np.ndarray):
    n = cm.shape[0]
    per_class_acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    precision = np.diag(cm) / cm.sum(axis=0).clip(min=1)
    recall = per_class_acc
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return per_class_acc, precision, recall, f1, pairs[:10]


def plot_per_class_accuracy(per_class_acc: np.ndarray, out_path: str):
    order = np.argsort(per_class_acc)
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=200)
    vals = per_class_acc[order] * 100.0
    labels = [CIFAR10_CLASSES[i] for i in order]
    bars = ax.barh(labels, vals, color="#4c78a8")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-class Accuracy (CIFAR-10 test)")
    ax.set_xlim(0, 100)
    for b, v in zip(bars, vals):
        ax.text(v + 0.7, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_reliability(labels: np.ndarray, preds: np.ndarray, confidences: np.ndarray, n_bins: int = 15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc_bin = np.zeros(n_bins, dtype=np.float64)
    conf_bin = np.zeros(n_bins, dtype=np.float64)
    count_bin = np.zeros(n_bins, dtype=np.int64)

    correct = (preds == labels).astype(np.float64)
    inds = np.digitize(confidences, bins[1:-1], right=True)
    for b in range(n_bins):
        mask = inds == b
        if mask.any():
            count_bin[b] = int(mask.sum())
            acc_bin[b] = float(correct[mask].mean())
            conf_bin[b] = float(confidences[mask].mean())

    ece = float(np.sum((count_bin / len(labels)) * np.abs(acc_bin - conf_bin)))
    return bins, acc_bin, conf_bin, count_bin, ece


def plot_reliability(acc_bin, conf_bin, count_bin, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=200)
    x = np.arange(len(acc_bin))
    ax.bar(x, acc_bin, width=0.8, alpha=0.8, color="#59a14f", label="accuracy")
    ax.plot(x, conf_bin, color="#e15759", marker="o", linewidth=1.5, label="confidence")
    ax.axhline(y=np.mean(acc_bin[count_bin > 0]) if np.any(count_bin > 0) else 0, color="gray", linestyle=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Value")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confidence_hist(confidences: np.ndarray, correct_mask: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=200)
    bins = np.linspace(0, 1, 30)
    ax.hist(confidences[correct_mask], bins=bins, alpha=0.65, label="correct", color="#4caf50")
    ax.hist(confidences[~correct_mask], bins=bins, alpha=0.65, label="incorrect", color="#f44336")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_timestep_ablation(model: SpikFormer, loader: DataLoader, timestep_values):
    original_t = model_def.TIME_STEP
    results = []
    for t in timestep_values:
        model_def.TIME_STEP = int(t)
        eval_res = evaluate(model, loader)
        results.append({"time_step": int(t), "accuracy": float(eval_res.accuracy)})
        print(f"[Timestep ablation] T={t}: acc={eval_res.accuracy:.4f}")
    model_def.TIME_STEP = original_t
    return results


def plot_timestep_ablation(ts_results, out_path):
    ts = [x["time_step"] for x in ts_results]
    acc = [x["accuracy"] * 100.0 for x in ts_results]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    ax.plot(ts, acc, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Inference time steps (T)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Time-step Ablation (same checkpoint)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_block_drop_ablation(model: SpikFormer, loader: DataLoader, baseline_acc: float):
    drops = []

    def _identity(self, x):
        return x

    for i, block in enumerate(model.blocks):
        original_forward = block.forward
        block.forward = types.MethodType(_identity, block)
        eval_res = evaluate(model, loader)
        block.forward = original_forward

        drop = baseline_acc - eval_res.accuracy
        drops.append({"block": i, "accuracy": float(eval_res.accuracy), "drop": float(drop)})
        print(f"[Block drop] remove block {i}: acc={eval_res.accuracy:.4f}, drop={drop:.4f}")
    return drops


def plot_block_drop_ablation(drop_results, out_path):
    blocks = [x["block"] for x in drop_results]
    drops = [x["drop"] * 100.0 for x in drop_results]
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=200)
    bars = ax.bar(blocks, drops, color="#ff9800")
    ax.set_xlabel("Dropped block index")
    ax.set_ylabel("Accuracy drop (pp)")
    ax.set_title("Block Ablation (single-block removal)")
    ax.set_xticks(blocks)
    for b, d in zip(bars, drops):
        ax.text(b.get_x() + b.get_width() / 2, d + 0.05, f"{d:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def collect_block_firing_rates(model: SpikFormer, loader: DataLoader):
    fired = [0.0 for _ in model.blocks]
    total = [0 for _ in model.blocks]
    handles = []

    def make_hook(block_idx):
        def hook(_, __, output):
            spikes = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(spikes):
                return
            spikes = spikes.detach()
            fired[block_idx] += spikes.sum().item()
            total[block_idx] += spikes.numel()
        return hook

    for i, block in enumerate(model.blocks):
        for mod in block.modules():
            if isinstance(mod, neuron.BaseNode):
                handles.append(mod.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            functional.reset_net(model)
            _ = model(x)

    for h in handles:
        h.remove()

    rates = [(f / t) if t > 0 else 0.0 for f, t in zip(fired, total)]
    overall = (sum(fired) / sum(total)) if sum(total) > 0 else 0.0
    return rates, overall


def plot_block_firing_rates(rates, overall, out_path):
    x = np.arange(len(rates))
    vals = np.array(rates) * 100.0
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=200)
    bars = ax.bar(x, vals, color="#4caf50")
    ax.axhline(overall * 100.0, color="red", linestyle="--", linewidth=1.5, label=f"overall {overall*100:.2f}%")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Firing rate (%)")
    ax.set_title("Average Spike Firing Rate per Block")
    ax.set_xticks(x)
    ax.legend()
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_markdown_report(path: str, summary: dict):
    lines = []
    lines.append("# SpikFormer CIFAR-10 Test-Set Analysis")
    lines.append("")
    lines.append(f"- **Checkpoint:** `{summary['checkpoint']}`")
    lines.append(f"- **Baseline test accuracy:** **{summary['baseline_accuracy']*100:.2f}%**")
    lines.append(f"- **ECE (15 bins):** **{summary['ece']:.4f}**")
    lines.append(f"- **Overall block firing rate:** **{summary['overall_firing_rate']*100:.3f}%**")
    lines.append("")
    lines.append("## Confusion and Class-wise Metrics")
    lines.append("")
    lines.append(f"- **Best class accuracy:** {summary['best_class']['name']} ({summary['best_class']['acc']*100:.2f}%)")
    lines.append(f"- **Worst class accuracy:** {summary['worst_class']['name']} ({summary['worst_class']['acc']*100:.2f}%)")
    lines.append("- **Top confusions (true -> predicted, count):**")
    for item in summary["top_confusions"][:5]:
        lines.append(f"  - {item['true']} -> {item['pred']}: {item['count']}")
    lines.append("")
    lines.append("## Ablations (test-set, no retraining)")
    lines.append("")
    lines.append("### Time-step ablation")
    lines.append("")
    lines.append("| T | Accuracy (%) |")
    lines.append("|---|---:|")
    for item in summary["timestep_ablation"]:
        lines.append(f"| {item['time_step']} | {item['accuracy']*100:.2f} |")
    lines.append("")
    lines.append("### Block drop ablation")
    lines.append("")
    lines.append("| Dropped block | Accuracy (%) | Drop (pp) |")
    lines.append("|---:|---:|---:|")
    for item in summary["block_drop_ablation"]:
        lines.append(f"| {item['block']} | {item['accuracy']*100:.2f} | {item['drop']*100:.2f} |")
    lines.append("")
    lines.append("### Block firing rates")
    lines.append("")
    lines.append("| Block | Firing rate (%) |")
    lines.append("|---:|---:|")
    for i, r in enumerate(summary["block_firing_rates"]):
        lines.append(f"| {i} | {r*100:.3f} |")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    for fig in summary["figures"]:
        lines.append(f"- `{fig}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Time-step and block-drop studies are **inference-time ablations** on a fixed checkpoint.")
    lines.append("- Time-step ablation changes the number of simulation steps without retraining.")
    lines.append("- Block-drop ablation replaces one transformer block with identity during inference.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "models/checkpoint_20260427-01h/"
            "best_spikformer_dim128_heads4_mlp256_layers4_epoch198_acc0.8551.pth"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="paper_figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(args.checkpoint)
    loader = get_test_loader(args.batch_size)

    # Baseline
    base = evaluate(model, loader)
    print(f"Baseline test accuracy: {base.accuracy:.4f}")

    # Confusion and class metrics
    cm = confusion_matrix(base.labels, base.preds)
    per_acc, precision, recall, f1, top_pairs = compute_class_metrics(cm)

    plot_confusion(cm, os.path.join(args.out_dir, "confusion_counts.png"), normalize=False)
    plot_confusion(cm, os.path.join(args.out_dir, "confusion_normalized.png"), normalize=True)
    plot_per_class_accuracy(per_acc, os.path.join(args.out_dir, "per_class_accuracy.png"))

    # Calibration/confidence
    _, acc_bin, conf_bin, count_bin, ece = compute_reliability(base.labels, base.preds, base.confidences, n_bins=15)
    plot_reliability(acc_bin, conf_bin, count_bin, os.path.join(args.out_dir, "reliability_diagram.png"))
    plot_confidence_hist(
        base.confidences,
        correct_mask=(base.preds == base.labels),
        out_path=os.path.join(args.out_dir, "confidence_histogram.png"),
    )

    # Ablations
    ts_results = evaluate_timestep_ablation(model, loader, timestep_values=[1, 2, 3, 4, 5, 6])
    plot_timestep_ablation(ts_results, os.path.join(args.out_dir, "timestep_ablation.png"))

    block_drop = evaluate_block_drop_ablation(model, loader, baseline_acc=base.accuracy)
    plot_block_drop_ablation(block_drop, os.path.join(args.out_dir, "block_drop_ablation.png"))

    block_rates, overall_rate = collect_block_firing_rates(model, loader)
    plot_block_firing_rates(block_rates, overall_rate, os.path.join(args.out_dir, "block_firing_rates.png"))

    best_idx = int(np.argmax(per_acc))
    worst_idx = int(np.argmin(per_acc))
    top_confusions = [
        {"true": CIFAR10_CLASSES[i], "pred": CIFAR10_CLASSES[j], "count": c}
        for i, j, c in top_pairs
    ]
    figures = [
        os.path.join(args.out_dir, "confusion_counts.png"),
        os.path.join(args.out_dir, "confusion_normalized.png"),
        os.path.join(args.out_dir, "per_class_accuracy.png"),
        os.path.join(args.out_dir, "reliability_diagram.png"),
        os.path.join(args.out_dir, "confidence_histogram.png"),
        os.path.join(args.out_dir, "timestep_ablation.png"),
        os.path.join(args.out_dir, "block_drop_ablation.png"),
        os.path.join(args.out_dir, "block_firing_rates.png"),
    ]

    summary = {
        "checkpoint": args.checkpoint,
        "baseline_accuracy": base.accuracy,
        "ece": ece,
        "overall_firing_rate": overall_rate,
        "block_firing_rates": block_rates,
        "best_class": {"name": CIFAR10_CLASSES[best_idx], "acc": float(per_acc[best_idx])},
        "worst_class": {"name": CIFAR10_CLASSES[worst_idx], "acc": float(per_acc[worst_idx])},
        "top_confusions": top_confusions,
        "timestep_ablation": ts_results,
        "block_drop_ablation": block_drop,
        "figures": figures,
    }

    json_path = os.path.join(args.out_dir, "analysis_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {json_path}")

    md_path = "paper_testset_findings.md"
    write_markdown_report(md_path, summary)
    print(f"Saved markdown report: {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
