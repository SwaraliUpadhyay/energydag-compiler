"""Operator DAG (a stand-in for an ONNX graph) and the rewrite-relevant metadata."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

KINDS = ["input", "conv", "dwconv", "bn", "relu", "add", "matmul",
         "softmax", "layernorm", "gelu", "pool", "fc"]
KIND_ID = {k: i for i, k in enumerate(KINDS)}
# Ops the (simulated) NPU cannot run natively -> CPU fallback penalty.
NPU_UNSUPPORTED = {"softmax", "layernorm", "gelu"}
# (producer, consumer) kind pairs that the fusion rewrite may merge.
FUSABLE_PAIRS = {("conv", "bn"), ("dwconv", "bn"), ("bn", "relu"), ("add", "relu"),
                 ("matmul", "gelu"), ("fc", "relu")}


@dataclass
class Op:
    name: str
    kind: str
    flops: float          # MFLOP
    in_mb: float          # fp32 activation MB read
    out_mb: float         # fp32 activation MB written
    params_mb: float
    preds: list = field(default_factory=list)
    sens: float = 0.1     # quantisation sensitivity (drives accuracy proxy)


class ModelGraph:
    def __init__(self, name: str):
        self.name = name
        self.ops: list[Op] = []

    # -- construction -------------------------------------------------
    def add(self, name, kind, preds, flops, out_mb, params_mb=0.0, sens=0.1) -> int:
        in_mb = sum(self.ops[p].out_mb for p in preds) if preds else out_mb
        self.ops.append(Op(f"{name}_{len(self.ops)}", kind, flops, in_mb, out_mb,
                           params_mb, list(preds), sens))
        return len(self.ops) - 1

    def finalize(self):
        n = len(self.ops)
        self.n = n
        self.flops = np.array([o.flops for o in self.ops])
        self.in_mb = np.array([o.in_mb for o in self.ops])
        self.out_mb = np.array([o.out_mb for o in self.ops])
        self.params_mb = np.array([o.params_mb for o in self.ops])
        self.sens = np.array([o.sens for o in self.ops])
        self.unsup = np.array([o.kind in NPU_UNSUPPORTED for o in self.ops])
        self.kind_id = np.array([KIND_ID[o.kind] for o in self.ops])
        cons = [0] * n
        for o in self.ops:
            for p in o.preds:
                cons[p] += 1
        self.n_consumers = cons
        edges = []
        for v, o in enumerate(self.ops):
            if len(o.preds) == 1:
                u = o.preds[0]
                if (self.ops[u].kind, o.kind) in FUSABLE_PAIRS and cons[u] == 1:
                    edges.append((u, v))
        self.fusable = edges                      # list[(u, v)], v ascending
        self.fuse_u = np.array([e[0] for e in edges], dtype=int)
        self.fuse_v = np.array([e[1] for e in edges], dtype=int)
        return self
