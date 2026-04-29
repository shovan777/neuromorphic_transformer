import torch
from torch import nn
import datetime
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional, neuron
from timm.data import mixup
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    set_seed,
    INPUT_CHANNELS,
    IMG_SIZE,
    MODEL_CONFIG,
    WARMUP_LR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    MIXUP_ALPHA,
    CUTMIX_ALPHA,
    USE_POISSON_ENCODING,
    DEBUG_MODE,
    DEBUG_EPOCHS,
    DEBUG_OVERFIT_STEPS,
    DEBUG_OVERFIT_BATCH_SIZE,
    RESUME_PATH,
    RESUME_EPOCH,
)
from dataset import get_cifar10_dataloaders
from model import SpikFormer


def save_model(model, path):
    torch.save(model.state_dict(), path)


def plot_training_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    loss_plot_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["test_accuracy"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    acc_plot_path = os.path.join(out_dir, "accuracy_curve.png")
    plt.savefig(acc_plot_path)
    plt.close()

    return loss_plot_path, acc_plot_path


def cupy_available():
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return True


class SpikeRateMonitor:
    def __init__(self, model):
        self.stats = defaultdict(lambda: {"fired": 0.0, "total": 0, "zeros": 0})
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, neuron.BaseNode):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_, __, output):
            spikes = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(spikes):
                return
            spikes = spikes.detach()
            self.stats[name]["fired"] += spikes.sum().item()
            self.stats[name]["total"] += spikes.numel()
            self.stats[name]["zeros"] += (spikes == 0).sum().item()
        return hook

    def reset(self):
        self.stats = defaultdict(lambda: {"fired": 0.0, "total": 0, "zeros": 0})

    def summary(self):
        out = {}
        for name, s in self.stats.items():
            if s["total"] == 0:
                continue
            out[name] = {
                "rate": s["fired"] / s["total"],
                "zero_frac": s["zeros"] / s["total"],
            }
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_overfit_probe(model, device, train_loader, optimizer, criterion, steps):
    data, target = next(iter(train_loader))
    data = data[:DEBUG_OVERFIT_BATCH_SIZE].to(device)
    target = target[:DEBUG_OVERFIT_BATCH_SIZE].to(device)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        functional.reset_net(model)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (step + 1) % 20 == 0 or step == 0:
            acc = (output.argmax(dim=1) == target).float().mean().item()
            print(f"[OVERFIT] step={step + 1}/{steps} loss={loss.item():.4f} acc={acc:.4f}")

def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    writer,
    criterion,
    soft_target_criterion,
    mixup_fn=None,
    spike_monitor=None,
):
    model.train()
    total_loss = 0.0
    train_acc = 0.0
    data_count = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        hard_target = target
        soft_target = None
        if mixup_fn is not None:
            data, soft_target = mixup_fn(data, target)
        optimizer.zero_grad()
        functional.reset_net(model)

        output = model(data)
        if soft_target is not None:
            loss = soft_target_criterion(output, soft_target)
        else:
            loss = criterion(output, hard_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * hard_target.numel()
        train_acc += (output.argmax(dim=1) == hard_target).sum().item()
        data_count += hard_target.numel()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_acc / data_count:.4f}"})
        
    avg_loss = total_loss / len(train_loader.dataset)
    train_accuracy = train_acc / data_count
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    spike_stats = spike_monitor.summary() if spike_monitor is not None else None
    return avg_loss, train_accuracy, spike_stats

def test(model, device, test_loader, epoch, writer, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    data_count = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch {epoch} [Test]", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            functional.reset_net(model)

            output = model(data)
            loss_val = criterion(output, target).item()
            test_loss += loss_val * target.numel()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            data_count += target.numel()
            
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{correct / data_count:.4f}"})

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", test_accuracy, epoch)
    return test_loss, test_accuracy

def main():
    set_seed(SEED)
    
    print("="*40)
    print("      TRAINING CONFIGURATION")
    print("="*40)
    print(f"Device:                {DEVICE}")
    print(f"Seed:                  {SEED}")
    print(f"Number of Epochs:      {NUM_EPOCHS}")
    print(f"Batch Size:            {BATCH_SIZE}")
    print(f"Learning Rate:         {LEARNING_RATE}")
    print(f"Warmup LR:             {WARMUP_LR}")
    print(f"Warmup Epochs:         {WARMUP_EPOCHS}")
    print(f"Weight Decay:          {WEIGHT_DECAY}")
    print(f"Stem Channels:         {MODEL_CONFIG['embed_dim']} (derived from embed_dim)")
    print(f"Embed Dim:             {MODEL_CONFIG['embed_dim']}")
    print(f"Num Heads:             {MODEL_CONFIG['num_heads']}")
    print(f"MLP Dim:               {MODEL_CONFIG['mlp_dim']}")
    print(f"Num Layers:            {MODEL_CONFIG['num_layers']}")
    print(f"Patch Size:            {MODEL_CONFIG['patch_size']}")
    print(f"Mixup Alpha:           {MIXUP_ALPHA}")
    print(f"CutMix Alpha:          {CUTMIX_ALPHA}")
    print(f"Poisson Encoding:      {USE_POISSON_ENCODING}")
    print(f"Debug Mode:            {DEBUG_MODE}")
    if DEBUG_MODE:
        print(f"Debug Epochs:          {DEBUG_EPOCHS}")
        print(f"Overfit Steps:         {DEBUG_OVERFIT_STEPS}")
        print(f"Overfit Batch Size:    {DEBUG_OVERFIT_BATCH_SIZE}")
    print("="*40)

    train_loader, test_loader, num_classes, class_names = get_cifar10_dataloaders(
        use_basic_augment=not DEBUG_MODE,
        use_random_augment=not DEBUG_MODE,
        random_erasing_prob=0.0 if DEBUG_MODE else 0.25,
        normalize=not USE_POISSON_ENCODING,
    )
    print(f"Number of classes: {num_classes}")
    print("Dataset: CIFAR10")

    if MODEL_CONFIG["embed_dim"] % MODEL_CONFIG["num_heads"] != 0:
        raise ValueError(
            f"embed_dim ({MODEL_CONFIG['embed_dim']}) must be divisible by "
            f"num_heads ({MODEL_CONFIG['num_heads']})."
        )
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
        use_cupy=torch.cuda.is_available() and cupy_available(),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_epochs = DEBUG_EPOCHS if DEBUG_MODE else NUM_EPOCHS
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_epochs,
    )
    mixup_fn = None
    criterion = nn.CrossEntropyLoss()
    soft_target_criterion = SoftTargetCrossEntropy()
    if not DEBUG_MODE and (MIXUP_ALPHA > 0 or CUTMIX_ALPHA > 0):
        mixup_fn = mixup.Mixup(
            mixup_alpha=MIXUP_ALPHA,
            cutmix_alpha=CUTMIX_ALPHA,
            num_classes=num_classes,
        )

    if DEBUG_MODE:
        run_overfit_probe(
            model=model,
            device=DEVICE,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            steps=DEBUG_OVERFIT_STEPS,
        )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model_size = num_params * 4 / (1024 * 1024) 
    print(f"Model size: {model_size:.2f} MB")

    writer = SummaryWriter(log_dir=f"runs/spikformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    spike_monitor = SpikeRateMonitor(model) if DEBUG_MODE else None
    checkpoint_date = time.strftime("%Y%m%d-%Hh")
    checkpoint_dir = f"models/checkpoint_{checkpoint_date}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_test_acc = -1.0
    best_epoch = -1
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    start_epoch = 1

    if RESUME_PATH is not None:
        if not os.path.isfile(RESUME_PATH):
            raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_PATH}")
        print(f"Resuming from: {RESUME_PATH} (epoch {RESUME_EPOCH + 1})")
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
        start_epoch = RESUME_EPOCH + 1
        # Fast-forward the LR scheduler to the correct position
        for _ in range(RESUME_EPOCH):
            lr_scheduler.step()

    for epoch in tqdm(range(start_epoch, train_epochs + 1), desc="Epochs"):
        epoch_idx = epoch - 1
        if (not DEBUG_MODE) and WARMUP_EPOCHS > 0 and epoch_idx < WARMUP_EPOCHS:
            lr_step = (LEARNING_RATE - WARMUP_LR) / WARMUP_EPOCHS
            current_lr = WARMUP_LR + lr_step * epoch_idx
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        train_loss, train_accuracy, spike_stats = train(
            model,
            DEVICE,
            train_loader,
            optimizer,
            epoch,
            writer,
            criterion,
            soft_target_criterion,
            mixup_fn=mixup_fn,
            spike_monitor=spike_monitor,
        )
        test_loss, test_accuracy = test(model, DEVICE, test_loader, epoch, writer, criterion)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], epoch)
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        if DEBUG_MODE and spike_stats:
            print("Spike diagnostics (rate | zero-frac):")
            for name in sorted(spike_stats.keys()):
                stats = spike_stats[name]
                print(f"  {name}: {stats['rate']:.6f} | {stats['zero_frac']:.6f}")
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch
            best_model_path = os.path.join(
                checkpoint_dir,
                f"best_spikformer_dim{MODEL_CONFIG['embed_dim']}_heads{MODEL_CONFIG['num_heads']}"
                f"_mlp{MODEL_CONFIG['mlp_dim']}_layers{MODEL_CONFIG['num_layers']}"
                f"_epoch{epoch}_acc{test_accuracy:.4f}.pth",
            )
            save_model(model, best_model_path)
            print(f"Saved best checkpoint: {best_model_path}")

        if (not DEBUG_MODE) and epoch_idx >= WARMUP_EPOCHS:
            lr_scheduler.step()

    final_model_path = os.path.join(
        checkpoint_dir,
        f"final_spikformer_dim{MODEL_CONFIG['embed_dim']}_heads{MODEL_CONFIG['num_heads']}"
        f"_mlp{MODEL_CONFIG['mlp_dim']}_layers{MODEL_CONFIG['num_layers']}"
        f"_epoch{train_epochs}_acc{history['test_accuracy'][-1]:.4f}.pth",
    )
    save_model(model, final_model_path)
    loss_plot_path, acc_plot_path = plot_training_curves(history, checkpoint_dir)
    print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"Saved final checkpoint: {final_model_path}")
    print(f"Saved plots: {loss_plot_path}, {acc_plot_path}")
    if spike_monitor is not None:
        spike_monitor.remove()
    writer.close()

if __name__ == "__main__":
    main()
    if spike_monitor is not None:
        spike_monitor.reset()
