import numpy as np
from energydag.zoo import ZOO
from energydag.plan import default_plan, build_exec, random_plan, mutate, canonicalize
from energydag.simulator import Simulator
from energydag.pareto import hv2d
from energydag.predictor import Ensemble
from energydag.search.problem import Problem
from energydag.search.active import active_search
from energydag.explain import explain


def test_graphs_are_dags():
    for f in ZOO.values():
        g = f()
        assert all(p < v for v, o in enumerate(g.ops) for p in o.preds)


def test_fusion_reduces_memory_traffic():
    g = ZOO["mobilenet"](); p = default_plan(g)
    q = p.copy(); q.fuse[:] = True
    assert build_exec(g, q).mem.sum() < build_exec(g, p).mem.sum()
    assert build_exec(g, q).n < build_exec(g, p).n


def test_simulator_deterministic_and_noise():
    g = ZOO["resnet"](); ex = build_exec(g, default_plan(g))
    assert Simulator.true_cost(ex) == Simulator.true_cost(ex)
    s = Simulator(seed=1)
    e, l = s.measure(ex); et, lt = Simulator.true_cost(ex)
    assert abs(e / et - 1) < 0.3


def test_hv2d():
    assert abs(hv2d([(0.5, 0.5)], (1, 1)) - 0.25) < 1e-12
    assert hv2d([(2, 2)], (1, 1)) == 0


def test_ensemble_learns():
    g = ZOO["pipeline"](); rng = np.random.default_rng(0); P = Problem(g); S = P.session(0)
    while S.n < 60:
        S.measure(random_plan(g, rng))
    m = Ensemble(3, epochs=200).fit(S.X, np.stack([S.E, S.L], 1))
    mu, sd = m.predict(np.array(S.X))
    assert np.abs(np.exp(mu[:, 0]) / np.array(S.E) - 1).mean() < 0.3 and (sd > 0).all()


def test_budget_respected_and_improves():
    P = Problem(ZOO["pipeline"]()); S = P.session(0)
    plan = active_search(P, S, 20)
    assert S.n == 20 and P.true_eval(plan)["E"] < 1.0


def test_explanation_deltas_sum_sign():
    g = ZOO["pipeline"](); P = Problem(g); plan = random_plan(g, np.random.default_rng(3))
    truth = lambda p: P.true_eval(p)["E"] * P.E0
    rows, _ = explain(g, plan, truth, truth)
    assert all(abs(r["delta_mJ"] - r["true_delta_mJ"]) < 1e-9 for r in rows)
