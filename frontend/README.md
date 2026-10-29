# EnergyDAG web UI

Dependency-free (Python stdlib + one HTML file) front end for the simulator.

    python frontend/server.py [--port 8000]
    # open http://127.0.0.1:8000

* **Optimise** tab: pick model / method / budget / constraints, run, and inspect the plan on the DAG
  (device = node colour, precision = ring colour, fused edges, cross-device transfers), convergence,
  and counterfactual explanations with predicted-vs-true deltas.
* **Benchmarks** tab: plots `results/results.csv` (produced by `make quick` / `make full`).

The server binds to 127.0.0.1 only. Runs are serialised (one at a time).
