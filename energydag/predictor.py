"""Uncertainty-aware energy/latency predictor: bootstrap ensemble of small MLPs (numpy only)
on top of SGC-style graph features. Std across members = epistemic uncertainty."""
from __future__ import annotations
import numpy as np


class _MLP:
    def __init__(self, d_in, d_out, hidden, rng):
        self.W1 = rng.normal(0, 1 / np.sqrt(d_in), (d_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, d_out)); self.b2 = np.zeros(d_out)

    def fwd(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def fit(self, X, Y, epochs=300, lr=0.01, l2=1e-3):
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        n = len(X)
        for t in range(1, epochs + 1):
            h, out = self.fwd(X)
            d = 2 * (out - Y) / n
            g2, gb2 = h.T @ d + l2 * self.W2, d.sum(0)
            dh = (d @ self.W2.T) * (1 - h ** 2)
            g1, gb1 = X.T @ dh + l2 * self.W1, dh.sum(0)
            for i, g in enumerate([g1, gb1, g2, gb2]):
                m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
                params[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)


class Ensemble:
    """Predicts (log energy, log latency). `predict` returns mean and std, each shape (n, 2)."""

    def __init__(self, k=5, hidden=48, epochs=300, seed=0):
        self.k, self.hidden, self.epochs, self.rng = k, hidden, epochs, np.random.default_rng(seed)

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.log(np.asarray(Y, float))
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-6
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-6
        Xn, Yn = (X - self.xm) / self.xs, (Y - self.ym) / self.ys
        self.members = []
        for _ in range(self.k):
            idx = self.rng.integers(0, len(X), len(X))            # bootstrap
            net = _MLP(X.shape[1], Y.shape[1], self.hidden, self.rng)
            net.fit(Xn[idx], Yn[idx], self.epochs)
            self.members.append(net)
        return self

    def predict(self, X):
        Xn = (np.asarray(X, float) - self.xm) / self.xs
        P = np.stack([m.fwd(Xn)[1] for m in self.members]) * self.ys + self.ym   # (k, n, 2) log-space
        return P.mean(0), P.std(0)
