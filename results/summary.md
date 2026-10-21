# Results (SIMULATED platform)

Metric: true energy of the recommended plan divided by the oracle plan's energy (1.00 = oracle; lower is better), mean over seeds, averaged over models. Default plan = 4.57x oracle.

| method | B=10 | B=25 | B=50 | B=100 |
|---|---|---|---|---|
| default | 4.57 | 4.57 | 4.57 | 4.57 |
| fused_best_device | 2.06 | 2.06 | 2.06 | 2.06 |
| random | 2.24 | 1.65 | 1.40 | 1.20 |
| latency_evo | 3.31 | 2.52 | 2.22 | 1.67 |
| evo_energy | 3.08 | 2.35 | 1.68 | 1.28 |
| offline_gnn | 2.55 | 1.48 | 1.09 | 0.97 |
| active (ours) | 2.80 | 1.43 | 1.11 | 0.85 |
| active, no uncertainty | 2.18 | 1.58 | 1.07 | 0.85 |
| active, GPU-only (no placement) | 1.51 | 1.27 | 1.24 | 1.23 |

## Predictor quality (60 random profiles, 200 held-out plans)

| model | energy MAPE | corr(std, abs err) |
|---|---|---|
| bert | 24.6% | 0.42 |
| mobilenet | 22.0% | -0.00 |
| pipeline | 25.0% | 0.41 |
| resnet | 32.8% | 0.28 |

## Explanation fidelity (predicted vs simulator-true per-decision deltas, oracle plan)

| model | pearson | spearman | #decisions |
|---|---|---|---|
| bert | 0.83 | 0.80 | 86 |
| mobilenet | 0.94 | 0.83 | 85 |
| pipeline | 0.77 | 0.91 | 60 |
| resnet | 0.64 | 0.73 | 78 |
