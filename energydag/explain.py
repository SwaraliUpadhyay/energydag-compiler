"""Counterfactual (leave-one-decision-out) energy explanations.

For every effective decision in a plan we revert *just that decision* to the stock default and report
the energy change attributable to it:  delta = E(plan) - E(plan with decision reverted)  (negative = saved).
Deltas are available from the learned predictor (cheap) and from the simulator (ground truth) so the
explanation's fidelity can be measured."""
from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr
from .plan import canonicalize, build_exec, PRECS, DEVICES


def decisions(graph, plan):
    q = canonicalize(graph, plan)
    ex = build_exec(graph, q)
    out = []
    for e in np.flatnonzero(q.fuse):
        out.append(("fuse", int(e), f"fuse {graph.ops[graph.fuse_u[e]].name} -> {graph.ops[graph.fuse_v[e]].name}"))
    heads = np.flatnonzero(ex.group == np.arange(graph.n))
    for h in heads:
        if q.prec[h] != 0:
            out.append(("prec", int(h), f"{graph.ops[h].name} precision -> {PRECS[q.prec[h]]}"))
        if q.dev[h] != 0:
            out.append(("dev", int(h), f"{graph.ops[h].name} placement -> {DEVICES[q.dev[h]]}"))
    return q, out


def revert(plan, kind, idx):
    r = plan.copy()
    if kind == "fuse":
        r.fuse[idx] = False
    elif kind == "prec":
        r.prec[idx] = 0
    else:
        r.dev[idx] = 0
    return r


def explain(graph, plan, energy_fn, truth_fn=None):
    """energy_fn(plan)->mJ. Returns list of dicts sorted by |delta|, plus fidelity if truth_fn given."""
    q, decs = decisions(graph, plan)
    base = energy_fn(q)
    base_t = truth_fn(q) if truth_fn else None
    rows = []
    for kind, idx, text in decs:
        r = revert(q, kind, idx)
        row = dict(decision=text, delta_mJ=base - energy_fn(r))
        if truth_fn:
            row["true_delta_mJ"] = base_t - truth_fn(r)
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["delta_mJ"]))
    fid = {}
    if truth_fn and len(rows) >= 3:
        a = [r["delta_mJ"] for r in rows]; b = [r["true_delta_mJ"] for r in rows]
        fid = dict(pearson=float(pearsonr(a, b)[0]), spearman=float(spearmanr(a, b)[0]), n=len(rows))
    return rows, fid
