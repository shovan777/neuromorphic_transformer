import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from config import DEVICE, NUM_CHANNELS, SEED, set_seed
from dataset import get_dataloaders
from model import SpikFormer
from spikingjelly.activation_based import functional

def visualize_attention_map_light():
    set_seed(SEED)
    
    print("Loading dataset...")
    train_loader, test_loader, num_classes, class_names = get_dataloaders()
    
    print("Loading model...")
    model = SpikFormer(num_classes=num_classes, num_channels=NUM_CHANNELS, use_cupy=True).to(DEVICE)
    
    # Normally you'd load a trained checkpoint here
    model.load_state_dict(torch.load("../spikformer_fmnist.pth", map_location=DEVICE))
    model.eval()

    # Get one batch of data
    data, target = next(iter(test_loader))
    data, target = data.to(DEVICE), target.to(DEVICE)

    # Need to reset network states!
    functional.reset_net(model)

    print("Running forward pass...")
    with torch.no_grad():
        output = model(data)

    # Extract attention weights from the last Block
    # Shape: [T, Batch, Heads, Patches, Patches] -> e.g., [8, 32, 4, 49, 49]
    attn_t = model.blocks[-1].attn.attn_weights
    
    print(f"Extracted attention weights shape: {attn_t.shape}")
    
    # --- Option 1: Extract attention at time step T=4 ---
    # attn_to_plot = attn_t[4]
    # title_suffix = "(T=4)"
    # save_filename = "attention_map_light_t4.png"

    # --- Option 2: Compute the mean over all time steps ---
    attn_to_plot = attn_t.mean(dim=0)
    title_suffix = "(Time Avg)"
    save_filename = "attention_map_light_avg.png"
    
    image_idx = 0
    num_heads = attn_to_plot.shape[1]
    
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(15, 4))
    
    # Plot original image
    img_np = data[image_idx].cpu().squeeze().numpy()
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title(f"Input: {class_names[target[image_idx].item()]}")
    axes[0].axis("off")
    
    # Plot attention map for each head as a binary plot (1 if spike/attn > 0 else 0)
    for head_idx in range(num_heads):
        # Shape: [Patches, Patches]
        attn_map = attn_to_plot[image_idx, head_idx].cpu().numpy()
        
        # Convert to binary: pixels where a spike occurred (attention > 0) are 1, else 0.
        attn_map_binary = (attn_map > 0).astype(int)
        
        # Using binary colormap (0=white, 1=black). We can also use 'gray' (0=black, 1=white)
        # Using interpolation="nearest" helps visualize distinct pixels/spikes
        im = axes[head_idx + 1].imshow(attn_map_binary, cmap="binary", interpolation="nearest")
        axes[head_idx + 1].set_title(f"Head {head_idx} Spikes {title_suffix}")
        axes[head_idx + 1].axis("off")
        
        # We don't really need a colorbar for binary, but keeping a simplified one for consistency
        fig.colorbar(im, ax=axes[head_idx + 1], fraction=0.046, pad=0.04, ticks=[0, 1])

    plt.tight_layout()
    
    # Save the output
    os.makedirs("visualizations", exist_ok=True)
    save_path = os.path.join("visualizations", save_filename)
    plt.savefig(save_path)
    print(f"Attention (Spikes) map saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_attention_map_light()
