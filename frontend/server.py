"""EnergyDAG web UI -- stdlib-only HTTP server (no extra dependencies).

    python frontend/server.py            # then open http://127.0.0.1:8000

API
  GET  /api/meta         models, methods, defaults
  GET  /api/graph?model= operator DAG of a model
  POST /api/run          run one search method on the simulator, return plan + explanation
  GET  /api/benchmarks   aggregated results/results.csv (if you have run `make quick` / `make full`)
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, threading, time, traceback
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
RESULTS = os.path.join(REPO, "results")
if REPO not in sys.path:                      # works even without `pip install -e .`
    sys.path.insert(0, REPO)

import numpy as np
from energydag.zoo import ZOO
from energydag.plan import PRECS, DEVICES, canonicalize, build_exec
from energydag.predictor import Ensemble
from energydag.explain import explain
from energydag.search.problem import Problem
from energydag.search.methods import METHODS
from energydag.search import baselines as B

BUDGET_FREE = ["default", "fused_best_device"]
RUN_LOCK = threading.Lock()                   # the oracle cache / numpy RNG work is not thread-safe


def clean(o):
    """Make anything JSON-safe (numpy scalars, NaN/inf -> null)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def meta():
    models = []
    for name, fn in ZOO.items():
        g = fn()
        models.append(dict(name=name, n_ops=g.n, n_fusable=len(g.fusable)))
    return dict(models=models, methods=list(METHODS), budget_free=BUDGET_FREE,
                devices=DEVICES, precisions=PRECS,
                defaults=dict(budget=50, seed=0, tol=1.0, slack=1.25))


def graph_json(model):
    if model not in ZOO:
        raise ValueError(f"unknown model {model!r}")
    g = ZOO[model]()
    ops = [dict(i=i, name=o.name, kind=o.kind, preds=o.preds, mflop=o.flops) for i, o in enumerate(g.ops)]
    return dict(model=model, ops=ops, fusable=[[int(u), int(v)] for u, v in g.fusable])


def run(req):
    model, method = req.get("model", "mobilenet"), req.get("method", "active (ours)")
    if model not in ZOO:
        raise ValueError(f"unknown model {model!r}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    budget = max(5, min(int(req.get("budget", 50)), 200))
    seed = int(req.get("seed", 0))
    tol, slack = float(req.get("tol", 1.0)), float(req.get("slack", 1.25))
    want_oracle = bool(req.get("oracle", False))

    P = Problem(ZOO[model](), tol=tol, slack=slack)
    S = P.session(seed)
    t0 = time.time()
    plan = METHODS[method](P, S, budget)
    elapsed = time.time() - t0
    r = P.true_eval(plan)
    g = P.graph

    q = canonicalize(g, plan)
    ex = build_exec(g, q)
    ops = [dict(i=i, name=o.name, kind=o.kind, dev=int(q.dev[i]), prec=int(q.prec[i]),
                group=int(ex.group[i]), fused=bool(ex.group[i] != i)) for i, o in enumerate(g.ops)]
    summary = dict(
        fused_groups=int(ex.n), ops=int(g.n), fused_edges=int(q.fuse.sum()),
        by_device={DEVICES[d]: int(sum(1 for o in ops if o["dev"] == d)) for d in range(3)},
        by_precision={PRECS[p]: int(sum(1 for o in ops if o["prec"] == p)) for p in range(3)})

    E = np.array(S.E) / S.E0
    L = np.array(S.L) / S.L0
    mask = S.feasible_mask()
    best, trace = 1.0, []
    for i in range(S.n):
        if mask[i]:
            best = min(best, float(E[i]))
        trace.append(best)
    measured = [dict(n=i + 1, E=float(E[i]), L=float(L[i]), feasible=bool(mask[i])) for i in range(S.n)]

    truth = lambda p: P.true_eval(p)["E"] * P.E0
    if S.n >= 10:
        model_ = Ensemble(5, seed=0).fit(S.X, np.stack([S.E, S.L], 1))
        pred = lambda p: float(np.exp(model_.predict(S.feat(p)[0][None])[0][0, 0]))
        rows, fid = explain(g, plan, pred, truth)
        source = "learned predictor (fit on this run's measurements)"
    else:
        rows, fid = explain(g, plan, truth, truth)
        source = "simulator ground truth (too few measurements to train a predictor)"

    out = dict(model=model, method=method, budget=budget, seed=seed, tol=tol, slack=slack,
               result=dict(E=r["E"], L=r["L"], acc=r["acc"], feasible=r["feasible"], E0_mJ=P.E0,
                           energy_mJ=r["E"] * P.E0),
               measurements=S.n, elapsed_s=elapsed, ops=ops, summary=summary,
               trace=trace, measured=measured, explanation=rows, fidelity=fid,
               explanation_source=source, stats=getattr(S, "stats", None))
    if want_oracle:
        _, orc = B.oracle(P)
        out["oracle"] = dict(E=orc["E"], L=orc["L"], acc=orc["acc"])
    return out


def benchmarks():
    path = os.path.join(RESULTS, "results.csv")
    if not os.path.exists(path):
        return dict(available=False)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["method"], int(r["budget"]))].append(float(r["E"]) / float(r["oracle_E"]))
    models = sorted({r["model"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    budgets = sorted({int(r["budget"]) for r in rows if int(r["budget"]) > 0})
    by_model = {m: defaultdict(dict) for m in models}
    for (m, me, b), v in agg.items():
        by_model[m][me][str(b)] = float(np.mean(v))
    overall = defaultdict(dict)
    for me in methods:
        for b in set(b for (_, mm, b) in agg if mm == me):
            vals = [by_model[m][me][str(b)] for m in models if str(b) in by_model[m][me]]
            overall[me][str(b)] = float(np.mean(vals))
    md_path = os.path.join(RESULTS, "summary.md")
    md = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""
    return dict(available=True, models=models, methods=methods, budgets=budgets,
                by_model=by_model, overall=overall, summary_md=md)


class Handler(BaseHTTPRequestHandler):
    server_version = "EnergyDAGUI/0.1"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(clean(body)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, e):
        traceback.print_exc()
        self._send(400, dict(error=f"{type(e).__name__}: {e}"))

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif u.path == "/api/meta":
                self._send(200, meta())
            elif u.path == "/api/graph":
                self._send(200, graph_json(parse_qs(u.query).get("model", ["mobilenet"])[0]))
            elif u.path == "/api/benchmarks":
                self._send(200, benchmarks())
            else:
                self._send(404, dict(error="not found"))
        except Exception as e:
            self._fail(e)

    def do_POST(self):
        try:
            if urlparse(self.path).path != "/api/run":
                return self._send(404, dict(error="not found"))
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            with RUN_LOCK:
                self._send(200, run(req))
        except Exception as e:
            self._fail(e)

    def log_message(self, fmt, *args):
        sys.stderr.write("[ui] " + (fmt % args) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"EnergyDAG UI running at http://{a.host}:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
