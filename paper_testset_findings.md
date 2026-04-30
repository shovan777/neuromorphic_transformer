# SpikFormer CIFAR-10 Test-Set Analysis

- **Checkpoint:** `models/checkpoint_20260427-01h/best_spikformer_dim128_heads4_mlp256_layers4_epoch198_acc0.8551.pth`
- **Baseline test accuracy:** **85.51%**
- **ECE (15 bins):** **0.1858**
- **Overall block firing rate:** **7.269%**

## Confusion and Class-wise Metrics

- **Best class accuracy:** automobile (95.30%)
- **Worst class accuracy:** cat (71.00%)
- **Top confusions (true -> predicted, count):**
  - dog -> cat: 113
  - cat -> dog: 93
  - deer -> bird: 63
  - cat -> bird: 51
  - cat -> frog: 51

## Ablations (test-set, no retraining)

### Time-step ablation

| T | Accuracy (%) |
|---|---:|
| 1 | 9.86 |
| 2 | 81.70 |
| 3 | 84.67 |
| 4 | 85.51 |
| 5 | 85.27 |
| 6 | 84.83 |

### Block drop ablation

| Dropped block | Accuracy (%) | Drop (pp) |
|---:|---:|---:|
| 0 | 73.75 | 11.76 |
| 1 | 80.08 | 5.43 |
| 2 | 79.16 | 6.35 |
| 3 | 81.97 | 3.54 |

### Block firing rates

| Block | Firing rate (%) |
|---:|---:|
| 0 | 7.860 |
| 1 | 7.548 |
| 2 | 7.829 |
| 3 | 5.840 |

## Generated Figures

- `paper_figures/confusion_counts.png`
- `paper_figures/confusion_normalized.png`
- `paper_figures/per_class_accuracy.png`
- `paper_figures/reliability_diagram.png`
- `paper_figures/confidence_histogram.png`
- `paper_figures/timestep_ablation.png`
- `paper_figures/block_drop_ablation.png`
- `paper_figures/block_firing_rates.png`

## Notes

- Time-step and block-drop studies are **inference-time ablations** on a fixed checkpoint.
- Time-step ablation changes the number of simulation steps without retraining.
- Block-drop ablation replaces one transformer block with identity during inference.
