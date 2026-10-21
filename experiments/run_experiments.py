"""Run all methods x models x budgets x seeds on the simulated platform; write CSV, summary, figures."""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np
from scipy.stats import spearmanr
from energydag.zoo import ZOO
from energydag.plan import random_plan, build_exec
from energydag.features import featurize
from energydag.predictor import Ensemble
from energydag.explain import explain
from energydag.search.problem import Problem
from energydag.search.methods import METHODS
from energydag.search import baselines as B

BUDGET_FREE = {"default", "fused_best_device"}


def predictor_quality(problem, n_train=60, n_test=200, seed=0):
    rng = np.random.default_rng(seed)
    S = problem.session(seed)
    while S.n < n_train:
        S.measure(random_plan(problem.graph, rng))
    m = Ensemble(5, seed=seed).fit(S.X, np.stack([S.E, S.L], 1))
    plans = [random_plan(problem.graph, rng) for _ in range(n_test)]
    F = np.array([S.feat(p)[0] for p in plans])
    truth = np.array([problem.true_eval(p) for p in plans], dtype=object)
    Et = np.array([t["E"] for t in truth]) * problem.E0
    mu, sd = m.predict(F)
    err = np.abs(np.exp(mu[:, 0]) - Et) / Et
    return dict(mape=float(err.mean()), unc_err_spearman=float(spearmanr(sd[:, 0], err)[0])), m, S


def explanation_fidelity(problem, model, S):
    plan, _ = B.oracle(problem)

    def pred(p):
        mu, _ = model.predict(S.feat(p)[0][None]); return float(np.exp(mu[0, 0]))

    truth = lambda p: problem.true_eval(p)["E"] * problem.E0
    return explain(problem.graph, plan, pred, truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 100])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, extra = [], {}
    t0 = time.time()
    for mname in a.models:
        P = Problem(ZOO[mname]())
        _, orc = B.oracle(P)
        q, model, S0 = predictor_quality(P)
        rows_f, fid = explanation_fidelity(P, model, S0)
        extra[mname] = dict(oracle_E=orc["E"], predictor=q, explanation_fidelity=fid,
                            example_explanation=rows_f[:8])
        for budget in a.budgets:
            for meth in a.methods:
                if meth in BUDGET_FREE and budget != a.budgets[0]:
                    continue
                for seed in range(a.seeds):
                    S = P.session(seed)
                    plan = METHODS[meth](P, S, budget)
                    r = P.true_eval(plan)
                    rows.append(dict(model=mname, method=meth, budget=0 if meth in BUDGET_FREE else budget,
                                     seed=seed, E=r["E"] if r["feasible"] else 1.0, L=r["L"], acc=r["acc"],
                                     feasible=r["feasible"], measurements=S.n, oracle_E=orc["E"]))
            print(f"[{time.time()-t0:5.0f}s] {mname} budget={budget} done", flush=True)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    json.dump(extra, open(os.path.join(a.out, "extra.json"), "w"), indent=2, default=float)
    from experiments.report import make_report
    make_report(a.out)


if __name__ == "__main__":
    main()
