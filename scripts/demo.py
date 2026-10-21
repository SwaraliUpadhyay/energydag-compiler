"""One-model demo: run the active search, print the plan and its counterfactual explanation."""
import argparse
import numpy as np
from energydag.zoo import ZOO
from energydag.plan import PRECS, DEVICES, canonicalize
from energydag.search.problem import Problem
from energydag.search.active import active_search
from energydag.predictor import Ensemble
from energydag.explain import explain
from energydag.search import baselines as B

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mobilenet", choices=list(ZOO))
ap.add_argument("--budget", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
P = Problem(ZOO[a.model]()); S = P.session(a.seed)
plan = active_search(P, S, a.budget)
r = P.true_eval(plan); _, orc = B.oracle(P)
print(f"{a.model}: {S.n} measurements | energy {r['E']:.2f}x default (oracle {orc['E']:.2f}x) | "
      f"latency {r['L']:.2f}x | acc drop {r['acc']:.2f}% | stats {S.stats}")
model = Ensemble(5, seed=0).fit(S.X, np.stack([S.E, S.L], 1))
pred = lambda p: float(np.exp(model.predict(S.feat(p)[0][None])[0][0, 0]))
rows, fid = explain(P.graph, plan, pred, lambda p: P.true_eval(p)["E"] * P.E0)
print("\nTop decisions (delta = energy change attributable, mJ; negative = saved):")
for x in rows[:10]:
    print(f"  {x['decision']:44s} predicted {x['delta_mJ']:+8.2f}   true {x['true_delta_mJ']:+8.2f}")
print("fidelity:", fid)
