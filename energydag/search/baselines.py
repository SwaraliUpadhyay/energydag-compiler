"""Baselines. Every method receives (problem, session, budget) and returns its recommended Plan."""
from __future__ import annotations
import numpy as np
from ..plan import Plan, random_plan, mutate, build_exec
from ..features import featurize
from ..predictor import Ensemble


def default(problem, S, budget):
    return problem.default


def fused_best_device(problem, S, budget):
    """ORT-like: fuse everything, fp32, pick best single device (3 measurements)."""
    g = problem.graph
    for d in problem.devices:
        S.measure(Plan(np.ones(len(g.fusable), bool), np.zeros(g.n), np.full(g.n, d)))
    return S.best()


def random_search(problem, S, budget):
    while S.n < budget:
        S.measure(random_plan(problem.graph, S.rng, problem.devices))
    return S.best()


def evo_measured(problem, S, budget, objective="energy"):
    """(mu+lambda)-style evolutionary search where every fitness evaluation is a measurement."""
    g, rng = problem.graph, S.rng
    n_init = max(5, budget // 5)
    while S.n < min(budget, n_init):
        S.measure(random_plan(g, rng, problem.devices))
    while S.n < budget:
        mask = S.feasible_mask()
        pool = np.flatnonzero(mask) if mask.any() else np.arange(S.n)
        vals = np.array(S.E if objective == "energy" else S.L)[pool]
        top = pool[np.argsort(vals)[:5]]
        for _ in range(20):
            child = (random_plan(g, rng, problem.devices) if rng.random() < 0.2
                     else mutate(g, S.plans[int(rng.choice(top))], rng, problem.devices))
            if child.key() not in S.keys:
                break
        S.measure(child)
    return S.best(objective)


# ---------- predicted-only search (shared by offline-GNN baseline and the oracle) -------------
def model_evo(problem, score_fn, rng, pop=40, gens=20, init=None):
    """score_fn(list[Plan]) -> (fitness array, lower is better). Returns list of (fitness, Plan) sorted."""
    g = problem.graph
    P = list(init or []) + [random_plan(g, rng, problem.devices) for _ in range(pop)]
    f = score_fn(P)
    for _ in range(gens):
        order = np.argsort(f)[:pop]
        P, f = [P[i] for i in order], f[order]
        kids = [mutate(g, P[int(rng.integers(len(P) // 2 + 1))], rng, problem.devices) for _ in range(pop)]
        P += kids
        f = np.concatenate([f, score_fn(kids)])
    order = np.argsort(f)
    return [(f[i], P[i]) for i in order]


def offline_gnn(problem, S, budget, k_verify=5, ensemble=5):
    """Offline surrogate (MileStone-style stand-in): random profiling -> train once -> optimise on
    predictions only -> verify top-k by measurement. No uncertainty, no active selection."""
    g, rng = problem.graph, S.rng
    while S.n < max(budget - k_verify, 6):
        S.measure(random_plan(g, rng, problem.devices))
    model = Ensemble(ensemble, seed=int(rng.integers(1 << 30))).fit(S.X, np.stack([S.E, S.L], 1))

    def score(plans):
        F, A = zip(*[S.feat(p) for p in plans])
        mu, _ = model.predict(np.array(F))
        e, l, a = np.exp(mu[:, 0]) / S.E0, np.exp(mu[:, 1]) / S.L0, np.array(A)
        return e + 10 * np.maximum(l - problem.slack, 0) + 10 * np.maximum(a - problem.tol, 0)

    ranked = model_evo(problem, score, rng, init=[S.best()])
    done = 0
    for _, p in ranked:
        if S.n >= budget or done >= k_verify:
            break
        if p.key() not in S.keys:
            S.measure(p); done += 1
    return S.best()


_ORACLE = {}


def oracle(problem, seed=0, pop=60, gens=150):
    """Large model-free search on the *noise-free* simulator: a (near-)optimal reference."""
    key = (problem.graph.name, problem.tol, problem.slack, problem.devices)
    if key in _ORACLE:
        return _ORACLE[key]
    rng = np.random.default_rng(seed)

    def score(plans):
        out = []
        for p in plans:
            r = problem.true_eval(p)
            out.append(r["E"] + 10 * max(r["L"] - problem.slack, 0) + 10 * max(r["acc"] - problem.tol, 0))
        return np.array(out)

    best = model_evo(problem, score, rng, pop=pop, gens=gens)[0][1]
    _ORACLE[key] = (best, problem.true_eval(best))
    return _ORACLE[key]
