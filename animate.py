import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=0.9)
ap.add_argument("--still", nargs=2, metavar=("FRAME", "OUT"))
ap.add_argument("--storyboard", action="store_true")
args = ap.parse_args()
TH = args.t
NF, POSTER = 540, 480

HERE = os.path.dirname(os.path.abspath(__file__))
for p in glob.glob(os.path.join(HERE, "media/fonts/*.ttf")):
    fm.fontManager.addfont(p)
PAPER, RAISED, INK, BODY, MUTED, RULE = "#f4f1ea", "#faf8f3", "#191713", "#3b362e", "#6d675e", "#b8b1a0"
RED = "#be241f"
SERIF, SANS, MONO = ["Instrument Serif", "DejaVu Sans"], ["IBM Plex Sans", "DejaVu Sans"], ["IBM Plex Mono", "DejaVu Sans"]
MONOM = ["IBM Plex Mono Medm", "IBM Plex Mono", "DejaVu Sans"]
PXW = 0.72
EPS = 1e-9

# ------------------------------------------------ data
sample = {}
for l in open(os.path.join(HERE, "data/sample.jsonl")):
    d = json.loads(l); sample[d["id"]] = d
N_TOTAL = len(sample)
items = []
for l in open(os.path.join(HERE, "results/decisions.jsonl")):
    d = json.loads(l)
    if d.get("status") != "ok" or d.get("probability") is None:
        continue
    s = sample[d["id"]]
    p = float(d["probability"])
    conf = round(max(p, 1 - p), 4)
    ans = p >= 0.5
    items.append(dict(id=d["id"], task=s["task"], text=" ".join(s["text"].split()), p=p, conf=conf,
                      yes=ans, correct=(ans == bool(s["human_label"])), sure=min(99, int(round(conf * 100)))))
N = len(items)
TAGS = {"cuad_anti-assignment": "CUAD · ANTI-ASSIGNMENT", "cuad_non-compete": "CUAD · NON-COMPETE",
        "cuad_change_of_control": "CUAD · CHANGE OF CONTROL", "hearsay": "HEARSAY",
        "personal_jurisdiction": "PERSONAL JURISDICTION", "consumer_contracts_qa": "CONSUMER CONTRACT"}
kept = [it for it in items if it["conf"] >= TH - EPS]
N_KEPT, N_KWRONG = len(kept), sum(not it["correct"] for it in kept)
ACC_KEPT = 100 * (N_KEPT - N_KWRONG) / N_KEPT
N_WRONG = sum(not it["correct"] for it in items)
SHARE = 100 * N_KEPT / N

NR = 42
wrong95 = [i for i, it in enumerate(items) if not it["correct"] and it["conf"] >= 0.95]
wrong_lo = [i for i, it in enumerate(items) if not it["correct"] and it["conf"] < 0.95]
right_all = [i for i, it in enumerate(items) if it["correct"]]


def pick_order():
    for seed in range(20000):
        rg = np.random.default_rng(seed)
        w95 = list(rg.choice(wrong95, 4, replace=False))
        wlo = list(rg.choice(wrong_lo, 3, replace=False))
        rt = list(rg.choice(right_all, NR - 7, replace=False))
        order = [int(i) for i in rg.permutation(w95 + wlo + rt)]
        last8 = order[-8:]
        lo = sum(items[i]["conf"] < TH - EPS for i in last8)
        cross_keep = sum((not items[i]["correct"]) and items[i]["conf"] >= TH - EPS for i in last8)
        tasks = [items[i]["task"] for i in order]
        run_ok = all(not (tasks[k] == tasks[k + 1] == tasks[k + 2]) for k in range(NR - 2))
        w95_tasks = len({items[i]["task"] for i in w95}) >= 3
        pos95 = sorted(order.index(i) for i in w95)
        if 2 <= lo <= 3 and cross_keep >= 1 and run_ok and w95_tasks and pos95[0] >= 6 and pos95[-1] <= 36:
            return order, seed
    raise SystemExit("no seed")


order42, SEED = pick_order()
rest = [i for i in range(N) if i not in set(order42)]
rg = np.random.default_rng(SEED + 1)
order_all = order42 + [int(i) for i in rg.permutation(rest)]
rows = [items[i] for i in order42]

# ------------------------------------------------ figure
fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1080); ax.set_ylim(1080, 0); ax.axis("off")
renderer = fig.canvas.get_renderer()

SX0, SX1, SY0, SY1 = 50, 1030, 185, 850
ROW0, RH = 226, 76
MIDY = ROW0 + (SY1 - ROW0) / 2
CX_NUM, CX_Q, CX_ANS, CX_SURE, CX_MARK = 68, 122, 698, 812, 984
sheet = Rectangle((SX0, SY0), SX1 - SX0, SY1 - SY0, facecolor=RAISED, edgecolor=RULE, linewidth=PXW, zorder=1)
ax.add_patch(sheet)
body = Rectangle((SX0, ROW0 + 1), SX1 - SX0, SY1 - ROW0 - 1, fill=False, edgecolor='none', zorder=0)
ax.add_patch(body)


def T(x, y, s, fam, size, color, ha="left", z=3, **kw):
    return ax.text(x, y, s, family=fam, fontsize=size, color=color, ha=ha, va="baseline", zorder=z, **kw)


def width(s, fam, size):
    t = ax.text(0, 0, s, family=fam, fontsize=size)
    w = t.get_window_extent(renderer).width
    t.remove()
    return w


def excerpt(text, maxw=555):
    text = " ".join(text.split()).lstrip(" .,;:-\u2013\u2014_\u2022\"'\u201c\u201d)")
    words = text.split(" ")
    cut = False
    while True:
        s = (" ".join(words).rstrip(" .,;:") + "…") if cut else " ".join(words)
        if width(s, SANS, 20) <= maxw or len(words) <= 3:
            return s
        words = words[:-1]; cut = True
        # target ~44 chars at most
        while len(" ".join(words)) > 44 and len(words) > 3:
            words = words[:-1]


for r in rows:
    r["exc"] = excerpt(r["text"]); r["tag"] = TAGS[r["task"]]

# header (static)
T(60, 52, f"A QUIZ WRITTEN BY LAWYERS · {N_TOTAL} QUESTIONS", MONO, 18, MUTED)
T(60, 108, "The model's answer sheet", SERIF, 40, INK)
T(60, 142, "CANDIDATE  gpt-6-luna      MARKED AGAINST  the lawyers' answers", MONO, 16, MUTED)
T(540, 1052, "OpenAI Decisions API · gpt-6-luna · LegalBench", MONO, 16, MUTED, ha="center")

# column headers + ink rule
colh = [T(CX_NUM, SY0 + 28, "#", MONO, 16, MUTED), T(CX_Q, SY0 + 28, "QUESTION", MONO, 16, MUTED),
        T(CX_ANS, SY0 + 28, "ANSWER", MONO, 16, MUTED), T(CX_SURE, SY0 + 28, "HOW SURE", MONO, 16, MUTED),
        T(CX_MARK, SY0 + 28, "MARK", MONO, 16, MUTED, ha="center")]
hrule, = ax.plot([SX0, SX0], [ROW0, ROW0], color=INK, lw=PXW, zorder=4, solid_capstyle="butt")

# row pool
POOL = 11
pool = []
for k in range(POOL):
    pool.append(dict(num=T(CX_NUM, 0, "", MONO, 16, MUTED), exc=T(CX_Q, 0, "", SANS, 20, BODY),
                     tag=T(CX_Q, 0, "", MONO, 16, MUTED), ans=T(CX_ANS, 0, "", MONOM, 24, INK),
                     sure=T(CX_SURE, 0, "", SERIF, 30, INK)))
for pr in pool:
    for t in pr.values():
        t.set_clip_path(body); t.set_clip_on(True)
rowrules = LineCollection([], colors=RULE, linewidths=PXW, zorder=2); ax.add_collection(rowrules)
rowrules.set_clip_path(body)
marks = LineCollection([], zorder=5, capstyle="round"); ax.add_collection(marks)
marks.set_clip_path(body)

# compact view (4 columns x 95 rows)
CC, CPAD, CGAP = 4, 25, 20
per_col = int(np.ceil(N_KEPT / CC))
cw = (SX1 - SX0 - 2 * CPAD - (CC - 1) * CGAP) / CC
pitch = (SY1 - SY0 - 2 * CPAD) / per_col
kept_order = [items[i] for i in order_all if items[i]["conf"] >= TH - EPS]
c_rule, c_tick, c_cross = [], [], []
for k, it in enumerate(kept_order):
    col, rw = k // per_col, k % per_col
    x0 = SX0 + CPAD + col * (cw + CGAP)
    y = SY0 + CPAD + rw * pitch + pitch / 2
    c_rule.append([(x0, y + pitch / 2), (x0 + cw, y + pitch / 2)])
    mx = x0 + cw - 10
    if it["correct"]:
        c_tick.append([(mx - 2.5, y), (mx - 1, y + 2)]); c_tick.append([(mx - 1, y + 2), (mx + 3, y - 2.5)])
    else:
        c_cross.append([(mx - 6, y - 6), (mx + 6, y + 6)]); c_cross.append([(mx + 6, y - 6), (mx - 6, y + 6)])
cv_rule = LineCollection(c_rule, colors=RULE, linewidths=PXW, zorder=2)
cv_tick = LineCollection(c_tick, colors=MUTED, linewidths=1.0 * PXW, zorder=3, capstyle="round")
cv_cross = LineCollection(c_cross, colors=RED, linewidths=2.0 * PXW, zorder=4, capstyle="round")
for c in (cv_rule, cv_tick, cv_cross):
    ax.add_collection(c)

# red-ink filter line
keep_line = T(SX0, 172, f"KEEP ONLY THE ANSWERS IT WAS {round(TH * 100)}% OR MORE SURE ABOUT", MONO, 16, INK)

# counters (footer area)
CXL, CXR = 60, 560
A = dict(ll=T(CXL, 892, "MARKED", MONO, 16, INK), ln=T(CXL, 972, "", SERIF, 64, INK),
         rl=T(CXR, 892, "WRONG", MONO, 16, INK), rn=T(CXR, 972, "", SERIF, 64, INK))
B = dict(ll=T(CXL, 892, "KEPT", MONO, 16, INK), ln=T(CXL, 972, "", SERIF, 64, INK),
         ls=T(CXL, 1006, f"{SHARE:.0f}% of the quiz", MONO, 16, MUTED),
         rl=T(CXR, 892, "MARKED WRONG", MONO, 16, MUTED), rn=T(CXR + 34, 972, "", SERIF, 64, INK))
bcross = LineCollection([[(CXR + 6, 940), (CXR + 22, 956)], [(CXR + 22, 940), (CXR + 6, 956)]], colors=RED,
                        linewidths=2 * PXW, zorder=4, capstyle="round")
ax.add_collection(bcross)
hl1 = T(60, 930, f"At {TH:g} confidence or more: {ACC_KEPT:.0f}% right on {SHARE:.0f}% of the quiz.", SERIF, 32, INK)
hl2 = T(60, 978, f"{N_KWRONG} answers it was sure about were marked wrong.", SANS, 21, BODY)


def clamp(v, a=0.0, b=1.0):
    return float(min(max(v, a), b))


def ease_io(p):
    p = clamp(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


P2A, P2B = 60, 270
TOTAL_SCROLL = (NR - 8) * RH


def scroll(f):
    x = clamp((f - P2A) / (P2B - P2A))
    return TOTAL_SCROLL * (0.35 * ease_io(x) + 0.65 * x)


# mark times
mark_t = []
for r in range(NR):
    t0 = None
    for f in range(P2A, P2B + 1):
        if ROW0 + r * RH + RH / 2 - scroll(f) <= MIDY:
            t0 = f; break
    if t0 is None:
        mark_t.append(250 + 5 * len([m for m in mark_t if m >= 250]))
    else:
        mark_t.append(P2A + 8 + 5 * r if t0 == P2A else t0 + 9)
mark_t = [max(m, P2A + 8) for m in mark_t]
removed = [r for r in range(NR - 8, NR) if rows[r]["conf"] < TH - EPS]


def draw_mark(cx, cy, correct, t, sc):
    if correct:
        pts = [(-9, 1), (-3, 8), (10, -9)]
        segs = [(pts[0], pts[1]), (pts[1], pts[2])]
        L = [np.hypot(8 - 0, 0) * 0 + np.hypot(6, 7), np.hypot(13, 17)]
        tot = sum(L); d = clamp(t) * tot
        out = []
        for (a, b_), l in zip(segs, L):
            if d <= 0:
                break
            fr = min(1, d / l); d -= l
            out.append(((cx + a[0] * sc, cy + a[1] * sc), (cx + (a[0] + (b_[0] - a[0]) * fr) * sc,
                                                           cy + (a[1] + (b_[1] - a[1]) * fr) * sc)))
        return out, [MUTED] * len(out), [2 * PXW * sc] * len(out)
    s1 = clamp(t * 2); s2 = clamp(t * 2 - 1)
    out, cols, wd = [], [], []
    for fr, a, b_ in ((s1, (-9, -9), (9, 9)), (s2, (9, -9), (-9, 9))):
        if fr > 0:
            out.append(((cx + a[0] * sc, cy + a[1] * sc), (cx + (a[0] + (b_[0] - a[0]) * fr) * sc,
                                                           cy + (a[1] + (b_[1] - a[1]) * fr) * sc)))
            cols.append(RED); wd.append(2.5 * PXW * sc)
    return out, cols, wd


def a_rgba(c, a):
    r_, g_, b_, _ = matplotlib.colors.to_rgba(c)
    return (r_, g_, b_, a)


def frame(f):
    L = clamp((f - 525) / 15)
    dyn = 1 - L
    # P1
    prog = clamp((f - 10) / 35)
    hrule.set_data([SX0, SX0 + (SX1 - SX0) * prog], [ROW0, ROW0]); hrule.set_alpha(dyn * (1 - clamp((f - 335) / 15)))
    cha = clamp((f - 30) / 25) * dyn * (1 - clamp((f - 335) / 15))
    for t in colh:
        t.set_alpha(cha)
    # zoom
    sc = 1 - 0.72 * ease_io((f - 335) / 40)
    big_a = (1 - clamp((f - 345) / 25)) * dyn
    cmp_a = clamp((f - 350) / 25) * dyn

    def Zx(x): return 540 + (x - 540) * sc
    def Zy(y): return MIDY + (y - MIDY) * sc

    appear = clamp((f - 55) / 15)
    sc_off = scroll(f)
    rr_segs, mk_segs, mk_cols, mk_w = [], [], [], []
    used = 0
    for pr in pool:
        for t in pr.values():
            t.set_alpha(0); t.set_text("")
    for r in range(NR):
        y0 = ROW0 + r * RH - sc_off
        a = appear
        xoff = 0.0
        if r in removed:
            a *= 1 - 0.8 * ease_io((f - 285) / 15)
            if f >= 298:
                q = ease_io((f - 298) / 20)
                xoff = -1000 * q; a *= 1 - q
        else:
            nrem = sum(1 for k in removed if k < r)
            y0 -= RH * nrem * ease_io((f - 315) / 20)
        a *= big_a
        if a <= 0.005 or y0 + RH < ROW0 - 5 or y0 > SY1 + 5 or used >= POOL:
            continue
        pr = pool[used]; used += 1
        d = rows[r]
        fs = sc
        def Pt(x, y): return Zx(x + xoff), Zy(y)
        for key, (x, dy, fam_size) in dict(num=(CX_NUM, 44, 16), exc=(CX_Q, 34, 20), tag=(CX_Q, 61, 16),
                                           ans=(CX_ANS, 46, 24), sure=(CX_SURE, 48, 30)).items():
            t = pr[key]
            px, py = Pt(x, y0 + dy)
            t.set_position((px, py)); t.set_fontsize(fam_size * fs); t.set_alpha(a)
        pr["num"].set_text(f"{r + 1}"); pr["exc"].set_text(d["exc"]); pr["tag"].set_text(d["tag"])
        pr["ans"].set_text("YES" if d["yes"] else "NO"); pr["sure"].set_text(f"{d['sure']}%")
        yb = Zy(y0 + RH)
        rr_segs.append([(Zx(SX0 + xoff), yb), (Zx(SX1 + xoff), yb)])
        mt = f - mark_t[r]
        if mt > 0:
            tt = mt / (4 if d["correct"] else 6)
            sg, cl, wd = draw_mark(Zx(CX_MARK + xoff), Zy(y0 + 38), d["correct"], tt, sc)
            for s_, c_, w_ in zip(sg, cl, wd):
                mk_segs.append(s_); mk_cols.append(a_rgba(c_, a)); mk_w.append(w_)
    rowrules.set_segments(rr_segs); rowrules.set_alpha(min(1.0, big_a * appear))
    marks.set_segments(mk_segs); marks.set_color(mk_cols if mk_cols else [RED]); marks.set_linewidths(mk_w if mk_w else [1])
    # compact
    for c in (cv_rule, cv_tick, cv_cross):
        c.set_alpha(cmp_a)
    # filter line
    ka = clamp((f - 270) / 20) * dyn
    keep_line.set_alpha(ka); keep_line.set_x(SX0 - 40 * (1 - ease_io((f - 270) / 20)))
    # counters
    ca = clamp((f - 60) / 20) * (1 - clamp((f - 375) / 20)) * dyn
    pe = ease_io((f - P2A) / (P2B - P2A))
    aa = 1 - clamp((f - 278) / 8)
    ab = clamp((f - 286) / 8)
    for k_, t in A.items():
        t.set_alpha(ca * aa)
    A["ln"].set_text(f"{round(N * pe)}"); A["rn"].set_text(f"{round(N_WRONG * pe)}")
    for k_, t in B.items():
        t.set_alpha(ca * ab)
    B["ls"].set_alpha(ca * clamp((f - 326) / 8))
    bcross.set_alpha(ca * ab)
    q = ease_io((f - 286) / 40)
    B["ln"].set_text(f"{round(N + (N_KEPT - N) * q)}"); B["rn"].set_text(f"{round(N_WRONG + (N_KWRONG - N_WRONG) * q)}")
    ha = clamp((f - 380) / 25) * dyn
    hl1.set_alpha(ha); hl2.set_alpha(ha)


def save_frame(f, path):
    frame(f)
    fig.savefig(path, dpi=100, facecolor=PAPER)


scratch = tempfile.mkdtemp(prefix="trustcurve_")
os.makedirs(os.path.join(HERE, "media"), exist_ok=True)


def storyboard():
    from PIL import Image
    idx = (150, 300, 480)
    ims = []
    for k in idx:
        pth = f"{scratch}/sb_{k}.png"
        save_frame(k, pth); ims.append(Image.open(pth).convert("RGB"))
    sb = Image.new("RGB", (3240, 1080), PAPER)
    for j, im in enumerate(ims):
        sb.paste(im, (j * 1080, 0))
    sb.save(os.path.join(HERE, "media/storyboard.png"))


if args.still:
    save_frame(int(args.still[0]), args.still[1])
elif args.storyboard:
    storyboard()
    print(f"n={N} kept={N_KEPT} wrong_kept={N_KWRONG} acc={ACC_KEPT:.1f} share={SHARE:.1f} wrong_all={N_WRONG} seed={SEED}")
    print("cross>=95:", [r["id"] for r in rows if (not r["correct"]) and r["conf"] >= 0.95])
else:
    fdir = f"{scratch}/frames"
    os.makedirs(fdir, exist_ok=True)
    t0 = time.time()
    for f in range(NF):
        save_frame(f, f"{fdir}/f{f:04d}.png")
    shutil.copy(f"{fdir}/f{POSTER:04d}.png", os.path.join(HERE, "media/trust_curve.png"))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{fdir}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    os.path.join(HERE, "media/trust_curve.mp4")], check=True)
    shutil.rmtree(fdir)
    print(f"rendered in {time.time() - t0:.0f}s")

shutil.rmtree(scratch, ignore_errors=True)
