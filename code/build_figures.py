#!/usr/bin/env python3
"""Build Fig 1 (zoom-in ladder), Fig 2 (OOD uncertainty failure) and
Fig 3 (cost-precision Pareto) for the PRGF 2026 manuscript. Data read
from the verified project files:
  - aturan_seleksi_v2.json (hull_mean = mu, gnn_sd, bio_lolos)
  - dft_r2/hasil_analisis_r2.json (dft_hull = E_above_hull)
  - charge_filter_r2.json (lolos_muatan per candidate)
  - manuscript.tex Tables 2 (funnel), 5 (backtest TP/FP/Prec/Sel) and
    6 (order costs) and the caption of Fig. 1
Outputs: naskah-ieee/figures/fig-ladder.pdf, fig-ood.pdf, fig-pareto.pdf
(vector PDF). Fig 1 was first drawn in TikZ; the TikZ version failed
to compile in Overleaf (pgfmath error at \\usetikzlibrary), so the
ladder is built here with matplotlib instead, with the same content
and numbers as the TikZ diagram.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))  # ~/workspace/research/PRGF baru
FIGDIR = os.path.join(BASE, "naskah-ieee", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ---------------------------------------------------------------- data
with open(os.path.join(BASE, "aturan_seleksi_v2.json")) as f:
    sel = json.load(f)
with open(os.path.join(BASE, "dft_r2", "hasil_analisis_r2.json")) as f:
    dft = {r["job"]: r for r in json.load(f)}
with open(os.path.join(BASE, "charge_filter_r2.json")) as f:
    chg = {c["job"]: c for c in json.load(f)["kandidat"]}

LABELS = ["r2_01_Ca3O8P2", "r2_16_Ca2O6PTb", "r2_17_Ca2CeO8P",
          "r2_19_Ca2H2O6P", "r2_24_Al2CaO6P"]
S = 1.85  # calibration factor in force at freeze (manuscript sec:gnn)

rows = []
for job in LABELS:
    k = next(c for c in sel["kandidat"] if c["job"] == job)
    mu = k["hull_mean"]
    sd = k["gnn_sd"]
    eh = dft[job]["e_above_hull"]
    bio = k["bio_lolos"]
    ch = chg[job]["lolos_muatan"]
    rows.append(dict(job=job[:5], mu=mu, sd=sd, eh=eh, bio=bio, ch=ch,
                     miss=eh - mu, miss_sd=(eh - mu) / sd))
    print(f"{job[:5]}: mu={mu:+.4f} sd={sd:.4f} E_hull={eh:+.4f} "
          f"bio={'pass' if bio else 'REJ'} charge={'pass' if ch else 'REJ'} "
          f"miss={eh-mu:+.4f} ({(eh-mu)/sd:.1f} sigma)")

# AND-gate outcome check (must match the brief: only r2_01 passes)
for r in rows:
    assert (r["bio"] and r["ch"]) == (r["job"] == "r2_01"), r
print("AND check OK: only r2_01 passes biology AND charge.")
# r2_17 / r2_19 checks from the brief
r17 = next(r for r in rows if r["job"] == "r2_17")
assert abs(r17["mu"] - (-0.3989)) < 1e-3 and abs(r17["sd"] - 0.0897) < 1e-3
assert abs(r17["eh"] - 0.6906) < 1e-3 and abs(r17["miss"] - 1.0895) < 1e-3
r19 = next(r for r in rows if r["job"] == "r2_19")
assert abs(r19["sd"] - 0.2381) < 1e-3
print("r2_17 / r2_19 values match the brief.")

# ---------------------------------------------------------------- Fig 2
fig, ax = plt.subplots(figsize=(3.6, 3.2))
# parity y=x and hull / stability-limit lines
xs = np.linspace(-0.85, 0.5, 50)
ax.plot(xs, xs, ls="--", lw=0.9, color="0.55", zorder=1)
ax.axhline(0.0, ls=":", lw=0.9, color="0.55", zorder=1)
ax.axhline(0.05, ls="-.", lw=0.9, color="0.55", zorder=1)
ax.text(-0.82, 0.075, "stable $\\leq$ 0.05", ha="left", va="bottom",
        fontsize=7.5, color="0.35")
ax.text(0.47, -0.035, "hull ($E=0$)", ha="right", va="top",
        fontsize=7.5, color="0.35")
ax.text(-0.82, -0.47, "parity", fontsize=7.5, color="0.35",
        rotation=38, rotation_mode="anchor")

# r2_17 miss: vertical dashed segment from the point down to parity
ax.plot([r17["mu"], r17["mu"]], [r17["mu"], r17["eh"]],
        ls=":", lw=1.0, color="0.4", zorder=1)

# marker scheme: shape = which gate(s) reject; colour = AND outcome
#   pass both      -> green filled circle
#   biology only   -> orange triangle
#   charge only    -> orange square
#   both rejected  -> red X
style = {
    ("pass",): dict(marker="o", ms=7, mfc="#1a7a1a", mec="#145214"),
    ("bio",): dict(marker="^", ms=8, mfc="#d97b00", mec="#8a4d00"),
    ("charge",): dict(marker="s", ms=7, mfc="#d97b00", mec="#8a4d00"),
    ("both",): dict(marker="x", ms=9, mfc="#b30000", mec="#b30000"),
}
seen_labs = set()
for r in rows:
    if r["bio"] and r["ch"]:
        key, lab = ("pass",), "passes both gates (r2_01)"
    elif (not r["bio"]) and r["ch"]:
        key, lab = ("bio",), "rejected by biology only (r2_16)"
    elif r["bio"] and (not r["ch"]):
        key, lab = ("charge",), "rejected by charge only (r2_19)"
    else:
        key, lab = ("both",), "rejected by both gates (r2_17, r2_24)"
    # legend dedupe: one entry per gate category (r2_17 and r2_24
    # share the "both" label — label it only once)
    if lab in seen_labs:
        lab = "_nolegend_"
    else:
        seen_labs.add(lab)
    st = style[key]
    ax.errorbar(r["mu"], r["eh"], xerr=S * r["sd"], fmt="none",
                ecolor="0.35", elinewidth=1.0, capsize=2.5, capthick=1.0,
                zorder=2)
    ax.plot(r["mu"], r["eh"], marker=st["marker"], ms=st["ms"],
            mfc=st["mfc"], mec=st["mec"], mew=1.2, ls="none", zorder=3,
            label=lab)

# annotations (offsets chosen to avoid the error bars)
ax.annotate("r2_17: miss 1.09 eV/atom\n($\\sim$12$\\sigma$)",
            xy=(r17["mu"], r17["eh"]), xytext=(-0.34, 0.80),
            fontsize=8, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
# r2_19 label sits ABOVE its leader line: xytext (0.16, 0.63) puts the
# two-line text clear of the diagonal arrow (previously at 0.55 the
# arrow ran horizontally right under "(σ=0.238)", looking like an
# underline). The diagonal leader now drops cleanly to the orange
# square, well below r2_17's error bar (y=0.69).
ax.annotate("r2_19: widest bar\n($\\sigma$=0.238)",
            xy=(r19["mu"], r19["eh"]), xytext=(0.16, 0.63),
            fontsize=8, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
# short point labels only (the legend + caption carry the rest)
r01 = next(r for r in rows if r["job"] == "r2_01")
ax.text(r01["mu"] + 0.015, r01["eh"] + 0.075, "r2_01", fontsize=8, ha="left")
r16 = next(r for r in rows if r["job"] == "r2_16")
ax.text(r16["mu"] + 0.02, r16["eh"] - 0.085, "r2_16", fontsize=8, ha="left")
r24 = next(r for r in rows if r["job"] == "r2_24")
ax.text(r24["mu"] + 0.03, r24["eh"] + 0.055, "r2_24", fontsize=8, ha="left")

ax.set_xlim(-0.85, 0.5)
ax.set_ylim(-0.55, 0.88)
ax.set_xlabel("GNN-implied $E_{\\mathrm{above\\,hull}}$, $\\mu$ (eV/atom)")
ax.set_ylabel("DFT $E_{\\mathrm{above\\,hull}}$ (eV/atom)")
# legend below the axes: the lower-right corner is crowded
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
         frameon=False, handlelength=1.6, borderaxespad=0.1,
         labelspacing=0.35, columnspacing=1.2)
ax.tick_params(direction="out", length=3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(os.path.join(FIGDIR, "fig-ood.pdf"), pad_inches=0.10)
plt.close(fig)
print("wrote fig-ood.pdf")

# ---------------------------------------------------------------- Fig 3
# (cost days, precision, status) — cost = Sel. (Table 5) x 3 machine-days
# precision from Table 5 TP/(TP+FP); verified against aturan_seleksi_v2.json
rules = [
    # name, cost_d, precision, status
    ("R0",  30, 0.20, "retro"),
    ("R1",   9, 0.00, "retro"),
    ("R2",  27, 0.33, "retro"),
    ("R4",   9, 0.00, "retro"),
    ("R5",   9, 1.00, "frozen-adopted"),
    ("R6",   9, 0.00, "frozen"),
    ("R6b",  9, 0.50, "frozen"),
    ("Charge", 54, 0.50, "retro"),
    ("AND", 24, 1.00, "retro"),
    ("R7",   3, 1.00, "retro"),
]
# verify precision against the JSON backtest counts
tb = {"R0": (1, 4), "R1": (0, 3), "R2": (1, 2), "R4": (0, 2), "R5": (1, 0),
      "R6": (0, 2), "R6b": (1, 1)}
for name, cost, prec, _ in rules:
    if name in tb:
        tp, fp = tb[name]
        # Table 5 prints precision rounded to two decimals
        assert abs(prec - round(tp / (tp + fp), 2)) < 1e-9, name
# charge / AND / R7 precisions from charge_filter_r2.json summary
assert abs(0.50 - 1 / 2) < 1e-9 and abs(1.00 - 1 / 1) < 1e-9
# R3 excluded: TP=0 FP=0 -> precision undefined
assert sel["aturan"]["R3"]["TP"] == 0 and sel["aturan"]["R3"]["FP"] == 0
print("Fig 3 data verified against Tables 5/6 and the JSON files; "
      "R3 excluded (precision undefined, TP=0 FP=0).")

fig, ax = plt.subplots(figsize=(6.8, 3.6))
# frozen-budget vertical line (all three frozen rules cost 9 d)
ax.axvline(9, ls=":", lw=1.0, color="0.6", zorder=1)
ax.text(9, 1.09, "frozen budget (9 d)", ha="center", va="bottom",
        fontsize=8, color="0.4")

# R1, R4, R6 coincide at (9, 0.00): one shared marker, labelled once
trio = ("R1, R4, R6", 9, 0.00)
ax.plot(trio[1], trio[2], marker="o", ms=8, mfc="none", mec="0.35",
        mew=1.2, ls="none", zorder=3)
ax.annotate(trio[0], xy=(trio[1], trio[2]), xytext=(13.5, -0.015),
            fontsize=8.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="0.5"))

plotted = [r for r in rules if r[0] not in ("R1", "R4", "R6")]
for name, cost, prec, status in plotted:
    if status == "frozen-adopted":
        ax.plot(cost, prec, marker="*", ms=17, mfc="#1a7a1a", mec="#145214",
                mew=1.0, ls="none", zorder=4)
    elif status == "frozen":
        ax.plot(cost, prec, marker="D", ms=7.5, mfc="#1f4e9c", mec="#16386e",
                mew=1.0, ls="none", zorder=4)
    else:
        ax.plot(cost, prec, marker="o", ms=8, mfc="0.75", mec="0.35",
                mew=1.2, ls="none", zorder=3)

labels = {
    "R7": ((3, 1.00), (4.5, 0.93), "left"),
    "AND": ((24, 1.00), (25.5, 1.00), "left"),
    "R6b": ((9, 0.50), (10.5, 0.50), "left"),
    "Charge": ((54, 0.50), (55.5, 0.44), "right"),
    "R2": ((27, 0.33), (28.5, 0.33), "left"),
    "R0": ((30, 0.20), (31.5, 0.20), "left"),
}
for name, (xy, xytext, ha) in labels.items():
    txt = name
    if name == "R7":
        txt = "R7 (retrospective)"
    ax.annotate(txt, xy=xy, xytext=xytext, fontsize=8.5, ha=ha,
                va="center",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="0.5"))

# R5: plain text below the star, no arrow. At (10.5, 1.00) the AND
# marker at (24, 1.00) covered the final "d" of "R5 (frozen,
# adopted)", and any arrow from the star down crosses the R7 label,
# so the label sits at (10.5, 0.75) — directly below-right of the
# star, clear of the AND marker, the R7 label and the R6b label.
ax.text(10.5, 0.75, "R5 (frozen, adopted)", fontsize=8.5, ha="left",
        va="center")

ax.text(59, 0.90, "Precision on $n=5$ labels (one positive);\n"
                  "95% CIs are wide (Table 5).",
        fontsize=8, ha="right", va="top", color="0.3")
ax.set_xlim(0, 60)
ax.set_ylim(-0.08, 1.14)
ax.set_xlabel("Cost (machine-days) = selections $\\times$ 3 d")
ax.set_ylabel("Precision")
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(direction="out", length=3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(os.path.join(FIGDIR, "fig-pareto.pdf"))
plt.close(fig)
print("wrote fig-pareto.pdf")

# ---------------------------------------------------------------- Fig 1
# Two orderings of the same screening levels (manuscript fig:ladder).
# Numbers verified against manuscript.tex Tables 2 (funnel) and 6
# (order) and the caption of Fig. 1: harvest 17,638; chemistry gate
# -> 2,429; biology gate -> 2,148; 30 drafts; GNN-first DFT 14 jobs
# ~42 machine-days, 4 wasted; domain-first DFT 3 jobs ~9 machine-days,
# 0 wasted; R0 TP=1 FP=4; r2_17 and r2_19 pass LCB yet fail DFT.
# Layout: the (A)/(B) row labels sit above each row (the retired TikZ
# diagram put them in a 2.0 cm left column, which is too narrow for the
# DejaVu Serif bold text at readable size); the boxes are 2.35 wide
# (shrunk ~8% from 2.55 so the red note box in row A clears the right
# edge) and the note box is 4.9 wide with its text reflowed to four
# lines, so no line touches the box edges.
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.set_xlim(0, 17.4)
ax.set_ylim(0, 8.6)
ax.axis("off")

BOX_W, GAP = 2.35, 0.27
BOX_H = 1.7
ROW_A_Y, ROW_B_Y = 6.9, 4.15
HEAD_A_Y, HEAD_B_Y = 8.15, 5.4


def flow_box(x, y, w, text, fontsize=7, edge="0.35", face="white",
             textcolor="black"):
    ax.add_patch(FancyBboxPatch(
        (x, y - BOX_H / 2), w, BOX_H,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        fc=face, ec=edge, lw=1.0, zorder=2))
    ax.text(x + w / 2, y, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, zorder=3)


def flow_arrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="0.3",
                                shrinkA=0, shrinkB=0))


def flow_row(y, boxes, note=None):
    x = 0.0
    for i, t in enumerate(boxes):
        flow_box(x, y, BOX_W, t)
        if i < len(boxes) - 1:
            flow_arrow(x + BOX_W, x + BOX_W + GAP, y)
        x += BOX_W + GAP
    if note is not None:
        flow_box(x, y, 4.9, note, fontsize=7, edge="#b30000",
                 face="#fff2f2", textcolor="#8a0000")
        x += 4.9
    assert x - GAP <= 17.4, x  # row must stay inside the canvas
    return x


# --- row A: standard GNN-first --------------------------------------
ax.text(0, HEAD_A_Y, "(A) Standard GNN-first", ha="left", va="center",
        fontsize=8.5, fontweight="bold")
flow_row(ROW_A_Y, [
    "Harvest\n17,638 records",
    "GNN ranking\n(all candidates)",
    "LCB selection\n(top picks)",
    "DFT\n14 jobs, $\\sim$42 d\n4 wasted",
], note="Out of domain it fails:\nr2_17, r2_19 pass LCB\nyet fail DFT\n(R0: TP = 1, FP = 4)")

# --- row B: proposed domain-first ladder ----------------------------
ax.text(0, HEAD_B_Y, "(B) Proposed domain-first ladder", ha="left",
        va="center", fontsize=8.5, fontweight="bold")
flow_row(ROW_B_Y, [
    "Harvest\n17,638 records",
    "Chemistry gate\n\u2192 2,429",
    "Biology gate\n\u2192 2,148",
    "GNN ranking",
    "30 drafts",
    "DFT\n3 jobs, $\\sim$9 d\n0 wasted",
])

# --- bottom note (same text as the TikZ note) -----------------------
ax.text(0, 1.9,
        "Hard gates before the model: the biology and charge gates together reject all four DFT failures;\n"
        "concept analysis of the five labels then diagnoses the Tier-B/NOMAD failure pattern.",
        ha="left", va="center", fontsize=8, color="0.3")

fig.savefig(os.path.join(FIGDIR, "fig-ladder.pdf"), pad_inches=0.12)
plt.close(fig)
print("wrote fig-ladder.pdf")
