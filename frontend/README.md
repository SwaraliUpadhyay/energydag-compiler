# EnergyDAG web UI

Dependency-free (Python stdlib + one HTML file) front end for the simulator.

    python frontend/server.py [--port 8000]
    # open http://127.0.0.1:8000

* **Optimise** tab: pick model / method / budget / constraints, run, and inspect the plan on the DAG
  (device = node colour, precision = ring colour, fused edges, cross-device transfers), convergence,
  and counterfactual explanations with predicted-vs-true deltas.
* **Benchmarks** tab: plots `results/results.csv` (produced by `make quick` / `make full`).

The server binds to 127.0.0.1 only. Runs are serialised (one at a time).

## Simulation pipeline view

Every run records the real steps taken (without modifying the library: `Session.measure`, `Ensemble.fit`
and `model_evo` are wrapped for the duration of the run). The UI shows:

* the 9-stage pipeline (DAG -> plan -> fusion -> simulator -> measure -> learn -> model search -> selection -> explain)
  with live counters from the run (measurements, ensemble fits, candidates scored on the model, skipped/explored picks);
* a replayable timeline (play / step / click a dot) that lights up the active stage and grows the convergence curve;
* a per-device Gantt chart of one inference (default vs recommended plan) with an energy breakdown
  (compute / link transfers / static). The schedule is re-played from `Simulator.true_cost` and cross-checked against it.
