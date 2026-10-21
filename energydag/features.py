"""Graph featurisation: per-node features -> SGC-style propagation -> pooled vector."""
from __future__ import annotations
import numpy as np

NB = 9                       # (device, precision) blocks
F_NODE = NB * 3 + 1 + 3     # block[flops, mem, 1] + npu-unsupported flag + small-op term per device
N_GLOBAL = 5


def featurize(ex) -> np.ndarray:
    M = ex.n
    X = np.zeros((M, F_NODE))
    b = ex.dev.astype(int) * 3 + ex.prec.astype(int)
    idx = np.arange(M)
    X[idx, b * 3 + 0] = ex.flops
    X[idx, b * 3 + 1] = ex.mem
    X[idx, b * 3 + 2] = 1.0
    X[:, NB * 3] = ((ex.dev == 2) & ex.unsup).astype(float)
    X[idx, NB * 3 + 1 + ex.dev.astype(int)] = np.exp(-ex.flops / 10.0)
    # one round of mean propagation over the undirected fused graph (with self loops)
    H = X.copy()
    cnt = np.ones(M)
    if len(ex.src):
        np.add.at(H, ex.src, X[ex.dst]); np.add.at(cnt, ex.src, 1)
        np.add.at(H, ex.dst, X[ex.src]); np.add.at(cnt, ex.dst, 1)
    H /= cnt[:, None]
    g = np.zeros(N_GLOBAL)
    if len(ex.src):
        cross = ex.dev[ex.src] != ex.dev[ex.dst]
        g[0] = ex.emb[cross].sum()
        g[1] = cross.sum()
        for k, (a, c) in enumerate([(0, 1), (0, 2), (1, 2)]):
            m = ((ex.dev[ex.src] == a) & (ex.dev[ex.dst] == c)) | ((ex.dev[ex.src] == c) & (ex.dev[ex.dst] == a))
            g[2 + k] = ex.emb[m].sum()
    return np.concatenate([X.sum(0), H.sum(0), g])
