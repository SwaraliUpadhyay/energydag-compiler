import numpy as np


def hv2d(points, ref):
    """Hypervolume of a 2-D minimisation point set w.r.t. reference point `ref`."""
    pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
    if not pts:
        return 0.0
    pts.sort()
    hv, best_l = 0.0, ref[1]
    for e, l in pts:
        if l < best_l:
            hv += (ref[0] - e) * (best_l - l)
            best_l = l
    return hv


def pareto_mask(P):
    P = np.asarray(P)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        dom = np.all(P <= P[i], 1) & np.any(P < P[i], 1)
        if dom.any():
            keep[i] = False
    return keep
