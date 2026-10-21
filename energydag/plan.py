"""A Plan = fusion bits + per-op precision + per-op device. Builds the executable (fused) graph."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PRECS = ["fp32", "fp16", "int8"]
DEVICES = ["cpu", "gpu", "npu"]


class Plan:
    __slots__ = ("fuse", "prec", "dev")

    def __init__(self, fuse, prec, dev):
        self.fuse = np.asarray(fuse, dtype=bool)
        self.prec = np.asarray(prec, dtype=np.int8)
        self.dev = np.asarray(dev, dtype=np.int8)

    def copy(self):
        return Plan(self.fuse.copy(), self.prec.copy(), self.dev.copy())

    def key(self):
        return self.fuse.tobytes() + self.prec.tobytes() + self.dev.tobytes()


def default_plan(graph) -> Plan:
    """What a stock runtime does: fp32, everything on the CPU, no fusion."""
    return Plan(np.zeros(len(graph.fusable), bool), np.zeros(graph.n, np.int8), np.zeros(graph.n, np.int8))


@dataclass
class ExecGraph:
    flops: np.ndarray      # per fused node, MFLOP
    mem: np.ndarray        # per fused node, MB moved through memory
    unsup: np.ndarray      # contains an NPU-unsupported op
    prec: np.ndarray
    dev: np.ndarray
    src: np.ndarray        # edges between fused nodes
    dst: np.ndarray
    emb: np.ndarray        # MB carried by each edge
    acc_loss: float        # accuracy-drop proxy (%), cheap and *not* a hardware measurement
    group: np.ndarray      # op -> root op id
    n: int


ACC_PEN = np.array([0.0, 0.05, 1.0])   # relative damage of fp32 / fp16 / int8 per unit sensitivity
ACC_FULL_INT8 = 4.0                    # % drop if *everything* were int8


def build_exec(graph, plan: Plan) -> ExecGraph:
    n = graph.n
    group = np.arange(n)
    for e in np.flatnonzero(plan.fuse):
        group[graph.fuse_v[e]] = group[graph.fuse_u[e]]
    roots = np.flatnonzero(group == np.arange(n))
    gid = np.full(n, -1)
    gid[roots] = np.arange(len(roots))
    g_of = gid[group]                                  # op -> fused node index
    m = len(roots)
    flops = np.bincount(g_of, graph.flops, m)
    params = np.bincount(g_of, graph.params_mb, m)
    tail = np.zeros(m, int)
    np.maximum.at(tail, g_of, np.arange(n))
    mem = graph.in_mb[roots] + graph.out_mb[tail] + params
    unsup = np.bincount(g_of, graph.unsup.astype(float), m) > 0
    prec, dev = plan.prec[roots], plan.dev[roots]
    src, dst, emb = [], [], []
    for v, op in enumerate(graph.ops):
        for p in op.preds:
            if g_of[p] != g_of[v]:
                src.append(g_of[p]); dst.append(g_of[v]); emb.append(graph.out_mb[p])
    acc = ACC_FULL_INT8 * float(np.sum(graph.sens * ACC_PEN[plan.prec[group]]) / graph.sens.sum())
    return ExecGraph(flops, mem, unsup, prec, dev, np.array(src, int), np.array(dst, int),
                     np.array(emb, float), acc, group, m)


def canonicalize(graph, plan: Plan) -> Plan:
    """Fused members inherit the head's precision/device (the only values that matter)."""
    ex = build_exec(graph, plan)
    q = plan.copy()
    q.prec = plan.prec[ex.group].copy()
    q.dev = plan.dev[ex.group].copy()
    return q


# ------------------------------------------------------------------ search-space operators
def random_plan(graph, rng, devices=(0, 1, 2), p_fuse=None) -> Plan:
    """Segment-structured random plan (contiguous runs share device/precision, like real placements)."""
    n = graph.n
    k = int(rng.integers(1, 7))
    cuts = np.sort(rng.integers(1, n, size=k - 1)) if k > 1 else np.array([], int)
    bounds = np.concatenate([[0], cuts, [n]])
    dev, prec = np.zeros(n, np.int8), np.zeros(n, np.int8)
    for a, b in zip(bounds[:-1], bounds[1:]):
        dev[a:b] = rng.choice(devices)
        prec[a:b] = rng.choice(3, p=[0.3, 0.35, 0.35])
    jit = rng.random(n) < 0.05
    prec[jit] = rng.integers(0, 3, jit.sum())
    pf = rng.random() if p_fuse is None else p_fuse
    fuse = rng.random(len(graph.fusable)) < pf
    return Plan(fuse, prec, dev)


def mutate(graph, plan: Plan, rng, devices=(0, 1, 2)) -> Plan:
    q = plan.copy()
    n = graph.n
    r = rng.random()
    if r < 0.35:                                       # segment mutation
        L = int(rng.integers(1, max(2, n // 3)))
        a = int(rng.integers(0, n - L + 1))
        what = rng.random()
        if what < 0.4 or what > 0.8:
            q.dev[a:a + L] = rng.choice(devices)
        if what >= 0.4:
            q.prec[a:a + L] = rng.integers(0, 3)
    elif r < 0.60:
        q.prec[int(rng.integers(n))] = rng.integers(0, 3)
    elif r < 0.75:
        q.dev[int(rng.integers(n))] = rng.choice(devices)
    elif len(q.fuse):
        for e in rng.integers(0, len(q.fuse), size=int(rng.integers(1, 4))):
            q.fuse[e] = ~q.fuse[e]
    return q
