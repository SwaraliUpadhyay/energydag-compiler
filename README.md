# EnergyDAG — uncertainty-aware, explainable energy optimisation of ML task-graphs

Runs **without hardware**: a deterministic, noisy-measurement *simulator* of a CPU/GPU/NPU edge platform
stands in for a power meter. All device constants are illustrative, not calibrated.

## Layout
```
energydag/
  graph.py, zoo.py      operator DAGs (mobilenet, resnet, bert, branching pipeline)
  plan.py               Plan = fusion bits + per-op precision + per-op device; fused ExecGraph; search operators
  simulator.py          ground-truth energy/latency + noisy "measurement" (swap for real harness)
  features.py           SGC-style graph features
  predictor.py          bootstrap-ensemble MLP -> mean + std (numpy only)
  search/               problem/session (budget accounting), baselines, active.py (ours), methods registry
  explain.py            counterfactual per-decision energy deltas + fidelity
experiments/            run_experiments.py, report.py
scripts/demo.py         one-model demo with explanation
tests/test_core.py
```
## Run
```
make install && make test
make demo          # single model, prints plan + explanation
make quick         # ~2 min sweep -> results/summary.md, energy_vs_budget.png
make full          # 4 models x budgets 10/25/50/100 x 5 seeds
```
## Honest status of the three claims (quick sweep: 3 models, 3 seeds; see results/summary.md)
1. **Active, measurement-budgeted search**: beats random/evo at B=25 and is close to the offline surrogate, but at
   B=50 the offline surrogate (1.04x oracle) is *better* than ours (1.20x). The "5-10x fewer measurements"
   claim is **not supported yet**. The uncertainty ablation (kappa=0) is not clearly worse. Ideas: better
   calibration, larger ensembles, tune kappa/batch/n_init, more seeds.
2. **Joint rewrite + placement**: the GPU-only ablation wins at tiny budgets (smaller space) while placement
   gives the best attainable plans (oracle) at larger budgets.
3. **Counterfactual explanations**: predicted vs true per-decision deltas correlate at 0.77-0.94 (Pearson).
   This is leave-one-decision-out, not Shapley; interactions are ignored.

Caveats: results are on a simulator whose structure the predictor features partly mirror, so they are
optimistic; accuracy is an analytic proxy; no real-hardware claim can be made until a real harness is plugged in.

## Going to real hardware
Implement `measure(ex)` in a class with the `Simulator` interface that runs the plan (ONNX Runtime/TensorRT)
and integrates power (INA219/tegrastats); keep `Session.measure` unchanged. Replace `true_cost` use in
`Problem` and the oracle with a long measured sweep.
