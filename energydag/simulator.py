"""Simulated heterogeneous edge platform = the stand-in for real power-meter measurements.

!! All constants are ILLUSTRATIVE (loosely shaped like Pi-CPU / Jetson-GPU / Coral-NPU class parts).
!! They are NOT calibrated to real silicon. Replace `Simulator` with a real harness to go live
!! (see README, section "Going to real hardware").
"""
from __future__ import annotations
import numpy as np
from .plan import build_exec

#             peak GFLOP/s by precision (fp32, fp16, int8), mem GB/s, static W, dynamic W, launch ms, knee MFLOP
DEV_SPECS = {
    0: dict(peak=(40, 32, 120),    bw=12, static=1.2, dyn=3.5, launch=0.03, knee=2.0),    # cpu
    1: dict(peak=(400, 900, 1200), bw=50, static=2.5, dyn=9.0, launch=0.08, knee=40.0),   # gpu
    2: dict(peak=(15, 150, 2000),  bw=16, static=0.4, dyn=2.0, launch=0.15, knee=20.0),   # npu
}
NPU_FALLBACK_SLOWDOWN = 8.0
LINK_GBPS, LINK_LAT_MS, LINK_MJ_PER_MB = 1.5, 0.04, 0.4


class Simulator:
    def __init__(self, sigma_e=0.04, sigma_l=0.02, n_rep=3, seed=0):
        self.sigma_e, self.sigma_l, self.n_rep = sigma_e, sigma_l, n_rep
        self.rng = np.random.default_rng(seed)

    # ---- ground truth (noise-free) -------------------------------------------------
    @staticmethod
    def true_cost(ex):
        """Returns (energy_mJ, latency_ms) of one inference of the fused graph."""
        M = ex.n
        finish = np.zeros(M)
        free = {0: 0.0, 1: 0.0, 2: 0.0}
        preds = [[] for _ in range(M)]
        for s, d, mb in zip(ex.src, ex.dst, ex.emb):
            preds[d].append((s, mb))
        dyn_e, used = 0.0, set()
        for i in range(M):
            d, p = int(ex.dev[i]), int(ex.prec[i])
            sp = DEV_SPECS[d]
            eff = 0.15 + 0.85 * ex.flops[i] / (ex.flops[i] + sp["knee"])
            comp = ex.flops[i] / (sp["peak"][p] * eff)
            mem_t = ex.mem[i] / sp["bw"]
            body = max(comp, mem_t)
            if d == 2 and ex.unsup[i]:
                body *= NPU_FALLBACK_SLOWDOWN
            util = comp / body if body > 0 else 0.0
            t = sp["launch"] + body
            dyn_e += sp["dyn"] * (0.3 * t + 0.7 * util * body)
            ready = 0.0
            for s, mb in preds[i]:
                tr = 0.0
                if ex.dev[s] != d:
                    tr = LINK_LAT_MS + mb / LINK_GBPS
                    dyn_e += LINK_MJ_PER_MB * mb
                ready = max(ready, finish[s] + tr)
            start = max(ready, free[d])
            finish[i] = start + t
            free[d] = finish[i]
            used.add(d)
        lat = float(finish.max())
        static_e = sum(DEV_SPECS[d]["static"] for d in used) * lat
        return dyn_e + static_e, lat

    # ---- what a power meter would report ------------------------------------------
    def measure(self, ex):
        e, l = self.true_cost(ex)
        k = np.sqrt(self.n_rep)
        return (e * float(np.exp(self.rng.normal(0, self.sigma_e / k))),
                l * float(np.exp(self.rng.normal(0, self.sigma_l / k))))

    def true_of_plan(self, graph, plan):
        ex = build_exec(graph, plan)
        e, l = self.true_cost(ex)
        return e, l, ex.acc_loss
