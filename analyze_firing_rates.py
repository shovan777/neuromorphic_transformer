import torch
import os
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from spikingjelly import visualizing
from model import SpikFormer
from config import DEVICE, NUM_CHANNELS, SEED, set_seed
from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.monitor import OutputMonitor

def analyze_firing_rates():
    set_seed(SEED)
    print("Loading dataset...")
    # Get a batch from the test loader
    _, test_loader, num_classes, _ = get_dataloaders()
    
    print("Loading model...")
    model = SpikFormer(num_classes=num_classes, num_channels=NUM_CHANNELS, use_cupy=True).to(DEVICE)

    
    # Normally you'd load a trained checkpoint here to analyze trained firing rates
    # e.g., model.load_state_dict(torch.load("checkpoint.pth"))
    model.load_state_dict(torch.load("../spikformer_fmnist.pth", map_location=DEVICE))
    model.eval()

    # 1. Attach OutputMonitor to all spiking neurons (BaseNode covers IFNode, LIFNode, etc.)
    monitor = OutputMonitor(model, neuron.BaseNode)

    # 2. Get a batch and run the forward pass
    data, target = next(iter(test_loader))
    data = data.to(DEVICE)
    
    print(f"\nRunning forward pass with batch size {data.size(0)}...\n")
    functional.reset_net(model)
    with torch.no_grad():
        output = model(data)

    # 3. Analyze and print firing rates
    print(f"{'Layer Name':<35} | {'Mean FR':<10} | {'Max FR':<10} | {'Zero FR (%)':<10}")
    print("-" * 73)
    
    for name in monitor.monitored_layers:
        spikes_list = monitor[name]
        if len(spikes_list) == 0:
            continue
            
        # spikes shape: [T, B, ...]
        # Note: SpikingJelly OutputMonitor caches lists of outputs for each forward pass.
        spikes = spikes_list[0]
        
        # Firing rate is the average spikes across the time dimension (T, which is dim 0)
        firing_rate = spikes.mean(dim=0)
        
        # Calculate statistics across the batch and spatial dimensions
        mean_fr = firing_rate.mean().item()
        max_fr = firing_rate.max().item()
        
        # Percentage of neurons that didn't fire a single spike
        zero_fr_pct = (firing_rate == 0).float().mean().item() * 100
        
        print(f"{name:<35} | {mean_fr:<10.4f} | {max_fr:<10.4f} | {zero_fr_pct:>5.1f}%")

    # 4. Visualize the spikes for the very first spiking layer (Img2Spike IFNode)
    print("\nGenerating spike visualizations...")
    # Get the records from the first spiking conv layer (e.g. img2spike.spike_conv.2)
    first_layer_name = 'img2spike.spike_conv.2'
    if first_layer_name in monitor.monitored_layers:
        img_spikes = monitor[first_layer_name][0] # Shape: [T, B, C, H, W]
        # We'll visualize the first image in the batch (index 0) over all time steps
        T = img_spikes.shape[0]
        C = img_spikes.shape[2]
        
        # Save visualizations to a folder
        os.makedirs("visualizations", exist_ok=True)
        print(f"Saving spike maps for {first_layer_name} to 'visualizations/' folder.")
        
        for t in range(T):
            # plotting max 8 channels per row
            visualizing.plot_2d_feature_map(
                img_spikes[t, 0].cpu().numpy(), 
                8, 
                max(1, C // 8), 
                2, 
                f"Spikes at $t={t}$"
            )
            plt.savefig(f"visualizations/img2spike_spikes_t{t}.png", pad_inches=0.02)
            plt.close()
    
    # 5. Clean up the hooks
    monitor.remove_hooks()
        
    print("\nAnalysis complete! Visualizations have been saved.")

if __name__ == "__main__":
    analyze_firing_rates()
