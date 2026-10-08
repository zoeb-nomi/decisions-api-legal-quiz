"""Calibration + trust curve for results/decisions.jsonl.

Writes results/metrics.json, results/trust_curve.csv, results/items.csv.
plot.py (separate) draws the static two-panel PNG from those files.
"""
import json, csv, os
from collections import defaultdict
import numpy as np

IN = "results/decisions.jsonl"
os.makedirs("results", exist_ok=True)

rows = [json.loads(l) for l in open(IN)]
# keep the latest record per id (judge.py appends on resume)
latest = {}
for r in rows: latest[r["id"]] = r
rows = list(latest.values())
ok = [r for r in rows if r.get("status") == "ok" and r.get("probability") is not None]
refusals = sum(r.get("status") == "refusal" for r in rows)
errors = sum(r.get("status") in ("error", "no_prob") for r in rows)
tokens = sum(r.get("input_tokens", 0) or 0 for r in rows)
usd = tokens / 1e6 * 0.10

def auroc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float)
    pos, neg = p[y], p[~y]
    if len(pos) == 0 or len(neg) == 0: return None
    # rank-based with tie handling (Mann-Whitney)
    order = np.argsort(p); ranks = np.empty(len(p)); 
    sp = p[order]; i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0; diag = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0: diag.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None}); continue
        conf, acc = p[m].mean(), y[m].mean()
        e += m.sum() / len(p) * abs(acc - conf)
        diag.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": float(conf), "acc": float(acc)})
    return float(e), diag

def metrics(rs):
    y = np.array([r["human_label"] for r in rs], bool)
    p = np.array([r["probability"] for r in rs], float)
    pred = p >= 0.5
    e, diag = ece(y, p)
    return {
        "n": len(rs),
        "pos_rate": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "auroc": auroc(y, p),
        "ece": e,
        "brier": float(((p - y) ** 2).mean()),
        "mean_p_given_true": float(p[y].mean()) if y.any() else None,
        "mean_p_given_false": float(p[~y].mean()) if (~y).any() else None,
        "reliability": diag,
    }

by_task = defaultdict(list)
for r in ok: by_task[r["task"]].append(r)
out = {"overall": metrics(ok), "per_task": {t: metrics(rs) for t, rs in sorted(by_task.items())},
       "refusals": refusals, "errors": errors, "input_tokens": tokens, "usd": round(usd, 4)}

# trust curve
y = np.array([r["human_label"] for r in ok], bool); p = np.array([r["probability"] for r in ok], float)
conf = np.maximum(p, 1 - p); correct = (p >= 0.5) == y
curve = []
for t in np.round(np.arange(0.50, 0.991, 0.01), 2):
    m = conf >= t
    curve.append({"threshold": float(t), "coverage": float(m.mean()),
                  "n": int(m.sum()), "accuracy": float(correct[m].mean()) if m.any() else None})
with open("results/trust_curve.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["threshold", "coverage", "n", "accuracy"]); w.writeheader(); w.writerows(curve)
with open("results/items.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "task", "human_label", "probability", "confidence", "correct"])
    for r, c, ok_ in zip(ok, conf, correct):
        w.writerow([r["id"], r["task"], int(r["human_label"]), r["probability"], round(float(c), 4), int(ok_)])

def at(t):
    m = conf >= t
    return {"threshold": t, "coverage": float(m.mean()), "n": int(m.sum()),
            "accuracy": float(correct[m].mean()) if m.any() else None,
            "wrong": int((~correct[m]).sum())}
worst = min(out["per_task"].items(), key=lambda kv: kv[1]["accuracy"])
# smallest threshold whose accuracy within the set reaches 0.95 / 0.99
def first_reaching(target):
    for c in curve:
        if c["accuracy"] is not None and c["accuracy"] >= target: return c
    return None
out["headline"] = {
    "overall_accuracy": out["overall"]["accuracy"],
    "at_0.9": at(0.9), "at_0.95": at(0.95), "at_0.99": at(0.99),
    "first_threshold_reaching_95pct": first_reaching(0.95),
    "first_threshold_reaching_99pct": first_reaching(0.99),
    "worst_task": {"task": worst[0], "accuracy": worst[1]["accuracy"], "auroc": worst[1]["auroc"]},
    "below_0.9_accuracy": float(correct[conf < 0.9].mean()) if (conf < 0.9).any() else None,
    "below_0.9_share": float((conf < 0.9).mean()),
}
json.dump(out, open("results/metrics.json", "w"), indent=2)

print(f"n={len(ok)} refusals={refusals} errors={errors} tokens={tokens} usd=${usd:.4f}")
print(f"{'task':28s} {'n':>4s} {'acc':>6s} {'auroc':>6s} {'ece':>6s} {'brier':>6s} {'p|yes':>6s} {'p|no':>6s}")
for t, m in list(out["per_task"].items()) + [("OVERALL", out["overall"])]:
    au = f"{m['auroc']:.3f}" if m['auroc'] is not None else "   na"
    print(f"{t:28s} {m['n']:4d} {m['accuracy']:6.3f} {au:>6s} {m['ece']:6.3f} {m['brier']:6.3f} {m['mean_p_given_true'] or 0:6.3f} {m['mean_p_given_false'] or 0:6.3f}")
h = out["headline"]
print(f"\nconf>=0.9: acc={h['at_0.9']['accuracy']:.3f} coverage={h['at_0.9']['coverage']:.3f} n={h['at_0.9']['n']} wrong={h['at_0.9']['wrong']}")
print(f"conf>=0.95: acc={h['at_0.95']['accuracy']:.3f} coverage={h['at_0.95']['coverage']:.3f}")
print(f"conf>=0.99: acc={(h['at_0.99']['accuracy'] or 0):.3f} coverage={h['at_0.99']['coverage']:.3f}")
print(f"conf<0.9: acc={h['below_0.9_accuracy']} share={h['below_0.9_share']:.3f}")
print(f"worst task: {h['worst_task']}")
