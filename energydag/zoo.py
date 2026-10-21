"""Synthetic-but-structurally-realistic model graphs (shapes -> FLOPs/MB are computed)."""
from __future__ import annotations
from .graph import ModelGraph


def _mb(h, w, c):  # fp32 activation size in MB
    return h * w * c * 4 / 1e6


class _B:
    def __init__(self, name):
        self.g = ModelGraph(name)

    def conv(self, x, h, w, cin, cout, k=1, kind="conv", sens=0.15):
        groups = cin if kind == "dwconv" else 1
        fl = 2 * h * w * cin * cout * k * k / groups / 1e6
        pm = cin * cout * k * k / groups * 4 / 1e6
        return self.g.add("conv" if kind == "conv" else "dw", kind, [x], fl, _mb(h, w, cout), pm, sens)

    def bn(self, x, h, w, c):
        return self.g.add("bn", "bn", [x], 4 * h * w * c / 1e6, _mb(h, w, c), 2 * c * 4 / 1e6, 0.1)

    def relu(self, x, h, w, c):
        return self.g.add("relu", "relu", [x], h * w * c / 1e6, _mb(h, w, c), 0, 0.05)

    def add(self, a, b, h, w, c):
        return self.g.add("add", "add", [a, b], h * w * c / 1e6, _mb(h, w, c), 0, 0.1)

    def cbr(self, x, h, w, cin, cout, k=1, kind="conv", relu=True, sens=0.15):
        y = self.bn(self.conv(x, h, w, cin, cout, k, kind, sens), h, w, cout)
        return self.relu(y, h, w, cout) if relu else y


def mobilenet_like(blocks=6):
    b = _B("mobilenet")
    x = b.g.add("input", "input", [], 0, _mb(224, 224, 3), 0, 0.0)
    h, c = 112, 32
    x = b.cbr(x, h, h, 3, c, 3, sens=1.0)
    cfg = [(32, 16, 1), (16, 24, 2), (24, 24, 1), (24, 32, 2), (32, 32, 1), (32, 64, 2),
           (64, 64, 1), (64, 96, 1)][:blocks]
    for cin, cout, s in cfg:
        inp, e = x, cin * 4
        y = b.cbr(x, h, h, cin, e, 1)
        h2 = h // s
        y = b.cbr(y, h2, h2, e, e, 3, "dwconv", sens=0.3)
        y = b.cbr(y, h2, h2, e, cout, 1, relu=False)
        x = b.add(inp, y, h2, h2, cout) if (s == 1 and cin == cout) else y
        h = h2
    x = b.cbr(x, h, h, cfg[-1][1], 256, 1)
    x = b.g.add("pool", "pool", [x], h * h * 256 / 1e6, _mb(1, 1, 256), 0, 0.1)
    b.g.add("fc", "fc", [x], 2 * 256 * 1000 / 1e6, _mb(1, 1, 1000), 256 * 1000 * 4 / 1e6, 1.0)
    return b.g.finalize()


def resnet_like():
    b = _B("resnet")
    x = b.g.add("input", "input", [], 0, _mb(224, 224, 3), 0, 0.0)
    x = b.cbr(x, 112, 112, 3, 64, 7, sens=1.0)
    x = b.g.add("pool", "pool", [x], 112 * 112 * 64 / 1e6, _mb(56, 56, 64), 0, 0.1)
    h, c = 56, 64
    for cout, blocks in [(64, 2), (128, 2), (256, 2)]:
        for i in range(blocks):
            s = 2 if (i == 0 and cout != c) else 1
            h2 = h // s
            y = b.cbr(x, h2, h2, c, cout, 3)
            y = b.cbr(y, h2, h2, cout, cout, 3, relu=False)
            skip = b.bn(b.conv(x, h2, h2, c, cout, 1), h2, h2, cout) if (s != 1 or c != cout) else x
            x = b.relu(b.add(skip, y, h2, h2, cout), h2, h2, cout)
            h, c = h2, cout
    x = b.g.add("pool", "pool", [x], h * h * c / 1e6, _mb(1, 1, c), 0, 0.1)
    b.g.add("fc", "fc", [x], 2 * c * 1000 / 1e6, _mb(1, 1, 1000), c * 1000 * 4 / 1e6, 1.0)
    return b.g.finalize()


def bert_tiny(layers=3, seq=128, d=256):
    g = ModelGraph("bert")
    act = _mb(seq, d, 1)
    mm = lambda n, p, k: g.add(n, "matmul", p, 2 * seq * d * k / 1e6, act, d * k * 4 / 1e6, 0.15)
    x = g.add("input", "input", [], 0, act, 0, 0.0)
    x = g.add("embed", "matmul", [x], 2 * seq * d * d / 1e6, act, d * d * 4 / 1e6, 1.0)
    for _ in range(layers):
        q, k, v = mm("q", [x], d), mm("k", [x], d), mm("v", [x], d)
        sc = g.add("scores", "matmul", [q, k], 2 * seq * seq * d / 1e6, _mb(seq, seq, 1), 0, 0.3)
        sm = g.add("softmax", "softmax", [sc], 5 * seq * seq / 1e6, _mb(seq, seq, 1), 0, 0.5)
        ctx = g.add("ctx", "matmul", [sm, v], 2 * seq * seq * d / 1e6, act, 0, 0.2)
        pr = mm("proj", [ctx], d)
        a1 = g.add("add", "add", [x, pr], seq * d / 1e6, act, 0, 0.1)
        n1 = g.add("ln", "layernorm", [a1], 8 * seq * d / 1e6, act, 0.002, 0.4)
        f1 = g.add("ffn1", "matmul", [n1], 2 * seq * d * 4 * d / 1e6, _mb(seq, 4 * d, 1), 4 * d * d * 4 / 1e6, 0.15)
        ge = g.add("gelu", "gelu", [f1], 8 * seq * 4 * d / 1e6, _mb(seq, 4 * d, 1), 0, 0.3)
        f2 = g.add("ffn2", "matmul", [ge], 2 * seq * 4 * d * d / 1e6, act, 4 * d * d * 4 / 1e6, 0.15)
        a2 = g.add("add", "add", [n1, f2], seq * d / 1e6, act, 0, 0.1)
        x = g.add("ln", "layernorm", [a2], 8 * seq * d / 1e6, act, 0.002, 0.4)
    g.add("fc", "fc", [x], 2 * seq * d * 2 / 1e6, _mb(seq, 2, 1), d * 2 * 4 / 1e6, 1.0)
    return g.finalize()


def pipeline():
    """Detector backbone -> two parallel heads (classifier, attribute) -> join. True branching DAG."""
    b = _B("pipeline")
    x = b.g.add("input", "input", [], 0, _mb(160, 160, 3), 0, 0.0)
    h, c = 80, 24
    x = b.cbr(x, h, h, 3, c, 3, sens=1.0)
    for cout in [32, 48, 64]:
        x = b.cbr(x, h // 2, h // 2, c, cout, 3)
        h, c = h // 2, cout
    feat = b.cbr(x, h, h, c, c, 3)
    heads = []
    for hn, depth in [("cls", 4), ("attr", 3)]:
        y = feat
        for _ in range(depth):
            y = b.cbr(y, h, h, c, c, 3)
        y = b.g.add("pool", "pool", [y], h * h * c / 1e6, _mb(1, 1, c), 0, 0.1)
        y = b.g.add("fc", "fc", [y], 2 * c * 128 / 1e6, _mb(1, 1, 128), c * 128 * 4 / 1e6, 1.0)
        heads.append(y)
    b.g.add("add", "add", heads, 128 / 1e6, _mb(1, 1, 128), 0, 0.1)
    return b.g.finalize()


ZOO = {"mobilenet": mobilenet_like, "resnet": resnet_like, "bert": bert_tiny, "pipeline": pipeline}
