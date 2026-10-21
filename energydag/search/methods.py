from functools import partial
from . import baselines as B
from .active import active_search

METHODS = {
    "default": B.default,
    "fused_best_device": B.fused_best_device,
    "random": B.random_search,
    "latency_evo": partial(B.evo_measured, objective="latency"),
    "evo_energy": B.evo_measured,
    "offline_gnn": B.offline_gnn,
    "active (ours)": active_search,
    "active, no uncertainty": partial(active_search, use_uncertainty=False),
    "active, GPU-only (no placement)": partial(active_search, devices=(1,)),
}
