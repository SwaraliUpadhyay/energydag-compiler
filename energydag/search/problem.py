"""Problem + measurement session: every call to Session.measure() costs one unit of budget."""
from __future__ import annotations
import numpy as np
from ..plan import default_plan, build_exec
from ..simulator import Simulator
from ..features import featurize


class Problem:
    def __init__(self, graph, tol=1.0, slack=1.25, devices=(0, 1, 2), seed=0):
        self.graph, self.tol, self.slack, self.devices, self.seed = graph, tol, slack, devices, seed
        self.default = default_plan(graph)
        self.E0, self.L0, _ = Simulator().true_of_plan(graph, self.default)   # noise-free reference

    def session(self, seed):
        return Session(self, seed)

    def true_eval(self, plan):
        e, l, a = Simulator().true_of_plan(self.graph, plan)
        return dict(E=e / self.E0, L=l / self.L0, acc=a,
                    feasible=bool(a <= self.tol and l / self.L0 <= self.slack * 1.02))


class Session:
    """Holds the measurement history. Accuracy proxy is free; energy/latency are 'hardware' calls."""

    def __init__(self, problem, seed):
        self.p, self.sim = problem, Simulator(seed=seed)
        self.rng = np.random.default_rng(seed + 12345)
        self.plans, self.E, self.L, self.acc, self.X, self.keys = [], [], [], [], [], set()
        self.measure(problem.default)                       # reference measurement (counts)
        self.E0, self.L0 = self.E[0], self.L[0]

    @property
    def n(self):
        return len(self.plans)

    def feat(self, plan):
        ex = build_exec(self.p.graph, plan)
        return featurize(ex), ex.acc_loss

    def measure(self, plan):
        ex = build_exec(self.p.graph, plan)
        e, l = self.sim.measure(ex)
        self.plans.append(plan); self.E.append(e); self.L.append(l); self.acc.append(ex.acc_loss)
        self.X.append(featurize(ex)); self.keys.add(plan.key())
        return e, l

    def feasible_mask(self):
        E, L, A = map(np.array, (self.E, self.L, self.acc))
        return (A <= self.p.tol) & (L <= self.p.slack * self.L0)

    def best(self, objective="energy"):
        mask = self.feasible_mask()
        if not mask.any():
            return self.p.default
        v = np.where(mask, np.array(self.E if objective == "energy" else self.L), np.inf)
        return self.plans[int(np.argmin(v))]

    def front_points(self):
        m = self.feasible_mask()
        return [(e / self.E0, l / self.L0) for e, l, k in zip(self.E, self.L, m) if k]
