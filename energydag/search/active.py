"""Measurement-budgeted active search (the core contribution).

Each round: fit ensemble -> model-only evolutionary search on the *optimistic* score
(LCB energy = mean - kappa*std, mean latency) -> keep a diverse shortlist -> measure only
candidates whose optimistic energy gain over the best measured feasible plan is positive
(confidently-not-better ones are skipped; if nothing qualifies, explore max-uncertainty) -> refit.
"""
from __future__ import annotations
import numpy as np
from ..plan import random_plan
from ..pareto import hv2d
from ..predictor import Ensemble
from .baselines import model_evo


def _dist(a, b):
    return int((a.dev != b.dev).sum() + (a.prec != b.prec).sum() + (a.fuse != b.fuse).sum())


def active_search(problem, S, budget, batch=4, kappa=1.0, ensemble=5, use_uncertainty=True,
                  devices=None, pop=40, gens=10, min_dist=4, log=None):
    g, rng = problem.graph, S.rng
    devices = tuple(devices or problem.devices)
    saved_devices = problem.devices
    problem.devices = devices                      # model_evo / mutate respect the allowed devices
    ref = (1.0, problem.slack)
    n_init = max(6, budget // 5)
    stats = dict(rounds=0, model_evals=0, skipped_confident=0, explore=0)
    try:
        while S.n < min(budget, n_init):
            S.measure(random_plan(g, rng, devices))
        while S.n < budget:
            model = Ensemble(ensemble, seed=int(rng.integers(1 << 30))).fit(S.X, np.stack([S.E, S.L], 1))
            k = kappa if use_uncertainty else 0.0
            cache = {}

            def predict(plans):
                F, A = zip(*[S.feat(p) for p in plans])
                mu, sd = model.predict(np.array(F))
                stats["model_evals"] += len(plans)
                for p, m, s, a in zip(plans, mu, sd, A):
                    cache[p.key()] = (m, s, a)
                return mu, sd, np.array(A)

            def score(plans):
                mu, sd, A = predict(plans)
                e = np.exp(mu[:, 0] - k * sd[:, 0]) / S.E0
                l = np.exp(mu[:, 1]) / S.L0
                return e + 10 * np.maximum(l - problem.slack, 0) + 10 * np.maximum(A - problem.tol, 0)

            mask = S.feasible_mask()
            order = np.argsort(np.where(mask, S.E, np.inf))[:5]
            init = [S.plans[i] for i in order if mask[i]]
            ranked = model_evo(problem, score, rng, pop=pop, gens=gens, init=init)
            best_e = min([S.E[i] / S.E0 for i in np.flatnonzero(mask)] + [1.0])

            short, seen = [], set()
            for f, p in ranked:
                if p.key() in S.keys or p.key() in seen:
                    continue
                seen.add(p.key()); short.append((f, p))
                if len(short) >= 40:
                    break
            gains = [max(0.0, best_e - f) for f, _ in short]
            chosen = []
            for i in np.argsort(gains)[::-1]:
                if gains[i] <= 1e-9 or len(chosen) >= min(batch, budget - S.n):
                    break
                if all(_dist(short[i][1], short[j][1]) >= min_dist for j in chosen):
                    chosen.append(int(i))
            stats["skipped_confident"] += sum(1 for gn in gains if gn <= 1e-9)
            while len(chosen) < min(batch, budget - S.n):        # explore: highest epistemic uncertainty
                cands = [random_plan(g, rng, devices) for _ in range(60)] + [p for _, p in short]
                mu, sd, A = predict(cands)
                u = np.where(A <= problem.tol, sd.sum(1), -1.0)
                for j in chosen:
                    u[len(cands) - len(short) + j] = -1.0
                short.append((np.inf, cands[int(np.argmax(u))])); chosen.append(len(short) - 1)
                stats["explore"] += 1
            for i in chosen:
                S.measure(short[i][1])
            stats["rounds"] += 1
            if log:
                log(S.n, hv2d(S.front_points(), ref))
    finally:
        problem.devices = saved_devices
    S.stats = stats
    return S.best()
