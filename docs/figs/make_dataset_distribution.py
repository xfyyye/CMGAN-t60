#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate two standalone T60 distribution figures from dataset_stats.txt.

Each figure is saved separately with its own complete axes (x + y labels):
  1. dataset_distribution_train.png — training pool (train + eval), blue bars.
  2. dataset_distribution_test.png  — test sets, grouped simulated vs measured RIR.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Parse dataset_stats.txt
# ---------------------------------------------------------------------------
STATS_FILE = Path(__file__).resolve().parents[2] / "dataset_stats.txt"
raw = STATS_FILE.read_text()

def _is_float(s):
    try:
        float(s); return True
    except ValueError:
        return False

def parse_block(header_key, ncols):
    rows = []
    lines = raw.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(header_key):
            for ln2 in lines[i + 1:]:
                parts = ln2.split()
                if len(parts) != ncols or not _is_float(parts[0]):
                    break
                rows.append([float(p) for p in parts])
            break
    return rows

# T60 rows: bin  train eval test1 test2 test3 test4  (7 cols)
t60 = np.array(parse_block("T60_BINS", 7))
bins = t60[:, 0]
cols = ["train", "eval", "test1", "test2", "test3", "test4"]
data = {c: t60[:, 1 + i] for i, c in enumerate(cols)}

train_pool = data["train"] + data["eval"]
sim_test   = data["test1"] + data["test3"]   # simulated RIR
real_test  = data["test2"] + data["test4"]   # measured RIR

# ---------------------------------------------------------------------------
# 2. Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "Arial",
    "font.size":        11,
    "axes.linewidth":   0.8,
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width":0.8,
    "ytick.major.width":0.8,
    "axes.grid":        True,
    "grid.color":       "#dddddd",
    "grid.linewidth":   0.6,
    "grid.linestyle":   "-",
})

BLUE   = "#2b6cb0"
ORANGE = "#dd6b42"

def style_axes(ax):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 3. Figure 1 — training pool T60 distribution
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6.0, 4.0))
ax1.bar(bins, train_pool, width=0.08, color=BLUE,
        edgecolor="white", linewidth=0.4, label="Train + Eval")
ax1.set_xlabel(r"Reverberation time $T_{60}$ (s)")
ax1.set_ylabel("Number")
ax1.set_xticks(bins)
ax1.set_xlim(0.05, 1.55)
ax1.set_ylim(0, train_pool.max() * 1.15)
ax1.yaxis.set_major_locator(MultipleLocator(2000))
style_axes(ax1)
fig1.tight_layout()
p1 = OUT / "dataset_distribution_train.png"
fig1.savefig(p1, dpi=300); plt.close(fig1)
print(f"saved -> {p1}  (total = {int(train_pool.sum())})")

# ---------------------------------------------------------------------------
# 4. Figure 2 — test-set T60 distribution, sim vs measured
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6.0, 4.0))
w = 0.045
ax2.bar(bins - w/2, sim_test,  width=w, color=BLUE,
        edgecolor="white", linewidth=0.4, label="Simulated RIR (test1 + test3)")
ax2.bar(bins + w/2, real_test, width=w, color=ORANGE,
        edgecolor="white", linewidth=0.4, label="Measured RIR (test2 + test4)")
ax2.set_xlabel(r"Reverberation time $T_{60}$ (s)")
ax2.set_ylabel("Number")
ax2.set_xticks(bins)
ax2.set_xlim(0.05, 1.55)
ax2.set_ylim(0, real_test.max() * 1.20)
ax2.yaxis.set_major_locator(MultipleLocator(100))
leg = ax2.legend(loc="upper right", frameon=True, framealpha=0.95,
                 edgecolor="#cccccc", fontsize=9, handlelength=1.4)
leg.get_frame().set_linewidth(0.6)
style_axes(ax2)
fig2.tight_layout()
p2 = OUT / "dataset_distribution_test.png"
fig2.savefig(p2, dpi=300); plt.close(fig2)
print(f"saved -> {p2}  (sim = {int(sim_test.sum())}, real = {int(real_test.sum())})")
