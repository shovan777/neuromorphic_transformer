# Experiment Results: SpikFormer CIFAR-10 Convergence Debugging

| Metric                    | Run 1 (no BN) | Run 2 (BN, v_th=1.0) | Run 3 (BN, v_th=0.5) |
|---------------------------|---------------|----------------------|----------------------|
| Overfit probe (200 steps) | 41%           | 34%                  | **95% ✅**           |
| Epoch 10 test acc         | 32.5%         | 28.9%                | 36.1%                |
| attn_lif firing rate      | ~0.003%       | ~1.5%                | **~10% ✅**          |
| fc2_lif firing rate       | 0% (dead)     | ~3%                  | **~14-18% ✅**       |

## Key Changes Per Run

- **Run 1 (no BN):** Baseline — no BatchNorm before LIF nodes in SSA/MLP. Dead neurons throughout, model unable to memorise a single batch.
- **Run 2 (BN, v_th=1.0):** Added BN before every LIF in SSA and MLP. Fixed dead `fc2_lif` but firing rates still too low; overfit probe regressed due to larger model (mlp_ratio=2).
- **Run 3 (BN, v_th=0.5):** Lowered threshold on all transformer LIF nodes to 0.5. Overfit probe passed at 95%, healthy firing rates across all blocks. Full 200-epoch run reached **85.51% test accuracy**.
