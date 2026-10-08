import csv
import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "media/fonts/*.ttf")):
    fm.fontManager.addfont(p)

PAPER, INK, BODY, MUTED, RULE, RED = "#f4f1ea", "#191713", "#3b362e", "#6d675e", "#b8b1a0", "#be241f"
SERIF, SANS, MONO = ["Instrument Serif", "DejaVu Sans"], ["IBM Plex Sans", "DejaVu Sans"], ["IBM Plex Mono", "DejaVu Sans"]
plt.rcParams.update({"font.family": SANS, "figure.facecolor": PAPER, "axes.facecolor": PAPER,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": RULE, "axes.spines.top": False, "axes.spines.right": False})

m = json.load(open("results/metrics.json"))
ov, h = m["overall"], m["headline"]["at_0.9"]
rows = list(csv.DictReader(open("results/trust_curve.csv")))
thr = [float(r["threshold"]) for r in rows]
cov = [100 * float(r["coverage"]) for r in rows]
acc_t = [float(r["threshold"]) for r in rows if r["accuracy"] != ""]
acc = [100 * float(r["accuracy"]) for r in rows if r["accuracy"] != ""]


def style(ax, xl, yl, title):
    ax.set_xlabel(xl.upper(), family=MONO, fontsize=9, color=MUTED, labelpad=8)
    ax.set_ylabel(yl.upper(), family=MONO, fontsize=9, color=MUTED, labelpad=8)
    ax.set_title(title, loc="left", family=SERIF, fontsize=20, color=INK, pad=12)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=0)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_family(MONO)


fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=200)
fig.suptitle(f"OpenAI Decisions API (gpt-6-luna) on LegalBench, n={ov['n']}", family=SERIF, fontsize=24, color=INK)

a.plot(thr, cov, color=MUTED, lw=2, label="coverage (auto-routable)")
a.plot(acc_t, acc, color=RED, lw=2, label="accuracy within that set")
a.axvline(0.9, color=INK, lw=0.9, ls="--")
a.plot([0.9], [100 * h["coverage"]], "o", color=INK, ms=7, mec=PAPER, mew=1.5)
a.plot([0.9], [100 * h["accuracy"]], "o", color=INK, ms=7, mec=PAPER, mew=1.5)
a.annotate(f"coverage {100 * h['coverage']:.0f}%", (0.9, 100 * h["coverage"]), xytext=(-8, -18),
           textcoords="offset points", ha="right", va="top", fontsize=11, color=INK, family=SANS)
a.annotate(f"accuracy {100 * h['accuracy']:.0f}%", (0.9, 100 * h["accuracy"]), xytext=(-8, 10),
           textcoords="offset points", ha="right", fontsize=11, color=INK, family=SANS)
a.text(0.9, 2, " THRESHOLD 0.9", fontsize=8, color=MUTED, ha="left", family=MONO)
a.set_xlim(0.5, 1.0); a.set_ylim(0, 105)
style(a, "confidence threshold", "%", "Trust curve")
leg = a.legend(loc="lower left", frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_family(SANS); t.set_color(BODY)

bins = [r for r in ov["reliability"] if r["n"] > 0]
b.bar([r["conf"] for r in bins], [100 * r["acc"] for r in bins], width=0.1, color=MUTED, alpha=0.9,
      edgecolor=PAPER, linewidth=1.5, label="observed per bin")
b.plot([0, 1.0], [0, 100], color=RULE, lw=1.6, ls="--", label="perfect calibration")
for r in bins:
    b.text(r["conf"], 100 * r["acc"] + 1.5, f"n={r['n']}", ha="center", fontsize=8, color=MUTED, family=MONO)
b.set_xlim(-0.05, 1.05); b.set_ylim(0, 110)
style(b, 'predicted probability of "yes"', 'observed share of "yes" (%)', "Reliability")
leg = b.legend(loc="upper left", frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_family(SANS); t.set_color(BODY)
b.text(0.03, 0.62, f"ECE {ov['ece']:.3f}\nBrier {ov['brier']:.3f}\nn = {ov['n']}", transform=b.transAxes,
       ha="left", va="top", fontsize=11, color=INK, family=SANS, linespacing=1.5)

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("results/trust_curve.png", facecolor=PAPER)
