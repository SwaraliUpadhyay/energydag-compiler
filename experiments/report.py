"""CSV -> summary.md + figures (no pandas dependency)."""
import csv, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(out):
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    for r in rows:
        r["budget"], r["E"], r["oracle_E"] = int(r["budget"]), float(r["E"]), float(r["oracle_E"])
        r["feasible"] = r["feasible"] == "True"
    return rows


def make_report(out):
    rows, extra = load(out), json.load(open(os.path.join(out, "extra.json")))
    models = sorted({r["model"] for r in rows}); methods = list(dict.fromkeys(r["method"] for r in rows))
    budgets = sorted({r["budget"] for r in rows if r["budget"] > 0})
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["method"], r["budget"])].append(r["E"] / r["oracle_E"])   # 1.0 == oracle

    def val(m, me, b):
        k = (m, me, 0 if me in ("default", "fused_best_device") else b)
        return np.mean(agg[k]) if agg.get(k) else np.nan

    lines = ["# Results (SIMULATED platform)\n", "Metric: true energy of the recommended plan divided by the oracle plan's energy "
             "(1.00 = oracle; lower is better), mean over seeds, averaged over models. Default plan = "
             f"{np.mean([1/extra[m]['oracle_E'] for m in models]):.2f}x oracle.\n",
             "| method | " + " | ".join(f"B={b}" for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for me in methods:
        lines.append(f"| {me} | " + " | ".join(f"{np.nanmean([val(m, me, b) for m in models]):.2f}" for b in budgets) + " |")
    lines += ["\n## Predictor quality (60 random profiles, 200 held-out plans)\n", "| model | energy MAPE | corr(std, abs err) |", "|---|---|---|"]
    for m in models:
        q = extra[m]["predictor"]; lines.append(f"| {m} | {q['mape']*100:.1f}% | {q['unc_err_spearman']:.2f} |")
    lines += ["\n## Explanation fidelity (predicted vs simulator-true per-decision deltas, oracle plan)\n", "| model | pearson | spearman | #decisions |", "|---|---|---|---|"]
    for m in models:
        f = extra[m]["explanation_fidelity"] or {}
        if f: lines.append(f"| {m} | {f['pearson']:.2f} | {f['spearman']:.2f} | {f['n']} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")

    fig, axs = plt.subplots(1, len(models), figsize=(4 * len(models), 3.4), squeeze=False)
    for ax, m in zip(axs[0], models):
        for me in methods:
            if me in ("default", "fused_best_device"):
                ax.axhline(val(m, me, 0), ls=":", lw=1, label=me, color="gray" if me == "default" else "black")
            else:
                ax.plot(budgets, [val(m, me, b) for b in budgets], marker="o", label=me)
        ax.set_yscale("log"); ax.set_title(m); ax.set_xlabel("measurements"); ax.set_ylabel("energy / oracle")
    axs[0][0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(out, "energy_vs_budget.png"), dpi=140); plt.close()
    print(open(os.path.join(out, "summary.md")).read())
