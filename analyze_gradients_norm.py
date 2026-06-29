"""
Gradient probe visualization for the THREE normalised-loss experiments.

Reports only the cosine between the two task gradients on the shared
encoder: the two-panel cos_r_bar.png (mean cosine + conflict fraction)
is the figure used in the paper (Fig. 3). The gradient magnitude-ratio
analysis was found to be confounded by the loss-normalisation and is
therefore not reported.

Usage:
    python analyze_gradients_norm.py [--out_dir figures/gradient_norm]

Figures:
    1. cos_r_bar.png     — 2-panel: mean cos, conflict fraction (paper Fig. 3)
    2. cos_hist.png      — cos distribution per regime
    3. cos_timeseries.png — encoder cos over training (smoothed)
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent

# Each experiment: (alpha, beta, rmse_ms, run_dir)
EXPERIMENTS = {
    "norm_w_denoise": {
        "log":   ROOT / "runs/kan_multitask_norm_w_denoise/nohup.log",
        "label": r"denoise-dom. ($\alpha$=1.0, $\beta$=0.1)",
        "short": "denoise-dom.",
        "color": "#2196F3",
        "alpha": 1.0, "beta": 0.1,
        "t60_rmse": 97.79,
    },
    "norm_balanced": {
        "log":   ROOT / "runs/kan_multitask_norm_balanced/nohup.log",
        "label": r"balanced ($\alpha$=1.0, $\beta$=1.0)",
        "short": "balanced",
        "color": "#4CAF50",
        "alpha": 1.0, "beta": 1.0,
        "t60_rmse": 109.90,
    },
    "norm_w_t60": {
        "log":   ROOT / "runs/kan_multitask_norm_w_t60/nohup.log",
        "label": r"$T_{60}$-dom. ($\alpha$=0.1, $\beta$=1.0)",
        "short": r"$T_{60}$-dom.",
        "color": "#F44336",
        "alpha": 0.1, "beta": 1.0,
        "t60_rmse": 109.54,
    },
}

# probe line; fields may be 'nan'. r is the probe's d/t ratio.
PROBE_RE = re.compile(
    r"\[probe step=(\d+)\]"
    r".*?encoder:cos=([+-]?nan|[+-]?\d+\.\d+),r=(nan|\d+\.\d+)"
    r".*?tscb_1:cos=([+-]?nan|[+-]?\d+\.\d+),r=(nan|\d+\.\d+)"
    r".*?tscb_2:cos=([+-]?nan|[+-]?\d+\.\d+),r=(nan|\d+\.\d+)"
)
LAYERS = ["encoder", "tscb_1", "tscb_2"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3,
})


def _f(s):
    try:
        v = float(s)
        return v if v == v else np.nan   # nan -> nan
    except ValueError:
        return np.nan


def parse_log(log_path, alpha, beta):
    """Return {layer: {'steps','cos','d_over_t'}}. probe r == d_over_t."""
    data = {l: {"steps": [], "cos": [], "d_over_t": []} for l in LAYERS}
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = PROBE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            vals = [
                (_f(m.group(2)), _f(m.group(3))),   # encoder
                (_f(m.group(4)), _f(m.group(5))),   # tscb_1
                (_f(m.group(6)), _f(m.group(7))),   # tscb_2
            ]
            for i, layer in enumerate(LAYERS):
                data[layer]["steps"].append(step)
                data[layer]["cos"].append(vals[i][0])
                data[layer]["d_over_t"].append(vals[i][1])
    for layer in LAYERS:
        for k in data[layer]:
            data[layer][k] = np.array(data[layer][k], dtype=float)
    return data


def effective_r(d_over_t, alpha, beta):
    """r_paper = ||beta*g_t60||/||alpha*g_den|| = (beta/alpha) * (1/(d/t)).

    Zero or non-finite probe ratios (e.g. r=0.00 / nan in the log) yield
    undefined r and are masked to nan so they are ignored by nanmean.
    """
    d_over_t = np.asarray(d_over_t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (beta / alpha) / d_over_t
    r = np.where((d_over_t <= 0) | ~np.isfinite(d_over_t), np.nan, r)
    return r


def smooth(arr, window=80):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    # ignore nan by using masked convolution
    mask = ~np.isnan(arr)
    out = np.full_like(arr, np.nan)
    if mask.sum() < window:
        out[mask] = arr[mask]
        return out
    kernel = np.ones(window) / window
    valid = np.where(mask, arr, 0.0)
    cnt = np.where(mask, 1.0, 0.0)
    s = np.convolve(valid, kernel, mode="same")
    c = np.convolve(cnt, kernel, mode="same")
    out = s / np.where(c == 0, 1, c)
    return out


def nanmean(a):
    a = np.asarray(a, dtype=float)
    return np.nanmean(a) if np.isfinite(a).any() else np.nan


# ── figures ──────────────────────────────────────────────────────────────────

def plot_bar_summary(all_data, out_dir):
    """Two-panel summary: (a) mean gradient cosine, (b) conflict fraction.

    Only cosine-based statistics are reported, since the magnitude-ratio
    analysis was found to be confounded by the loss-normalisation and is
    therefore not used in the paper.
    """
    exps = list(all_data.keys())
    cfgs = [EXPERIMENTS[e] for e in exps]
    labels = [c["short"] for c in cfgs]
    colors = [c["color"] for c in cfgs]

    cos_means = [nanmean(all_data[e]["encoder"]["cos"]) for e in exps]
    neg_pcts = [np.nanmean(all_data[e]["encoder"]["cos"][~np.isnan(all_data[e]["encoder"]["cos"])] < 0) * 100 for e in exps]

    x = np.arange(len(exps))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) mean cos
    bars = ax_a.bar(x, cos_means, color=colors, edgecolor="white", linewidth=0.6)
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel("Mean gradient cosine")
    ax_a.set_title("(a) Mean cosine between task gradients")
    for b, v in zip(bars, cos_means):
        ax_a.text(b.get_x()+b.get_width()/2, v+0.001, f"{v:+.3f}",
                  ha="center", va="bottom", fontsize=9)
    ax_a.set_ylim(0, 0.045)

    # (b) neg% (conflict fraction)
    bars = ax_b.bar(x, neg_pcts, color=colors, edgecolor="white", linewidth=0.6)
    ax_b.axhline(50, color="red", linewidth=0.8, linestyle="--", alpha=0.6, label="50% (random)")
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels, fontsize=9)
    ax_b.set_ylabel(r"Conflict fraction ($\cos<0$)")
    ax_b.set_title("(b) Fraction of conflicting steps")
    ax_b.set_ylim(0, 65); ax_b.legend(fontsize=8)
    for b, v in zip(bars, neg_pcts):
        ax_b.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.1f}%",
                  ha="center", va="bottom", fontsize=9)

    fig.suptitle("Gradient-probe analysis across the three normalised regimes",
                 fontsize=13, fontweight="bold", y=1.02)
    path = out_dir / "cos_r_bar.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cos_hist(all_data, out_dir):
    exps = list(all_data.keys())
    fig, axes = plt.subplots(1, len(exps), figsize=(4.2*len(exps), 4), sharey=True)
    if len(exps) == 1:
        axes = [axes]
    for ax, e in zip(axes, exps):
        cfg = EXPERIMENTS[e]
        cos = all_data[e]["encoder"]["cos"]
        cos = cos[~np.isnan(cos)]
        ax.hist(cos, bins=50, color=cfg["color"], alpha=0.85,
                edgecolor="white", linewidth=0.3, density=True)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        mu = np.mean(cos)
        ax.axvline(mu, color="darkred", linewidth=1.5, label=f"mean={mu:+.3f}")
        neg = np.mean(cos < 0) * 100
        ax.set_title(f"{cfg['short']}\nneg%={neg:.1f}%", fontsize=10)
        ax.set_xlabel("cos similarity"); ax.set_xlim(-1, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    path = out_dir / "cos_hist.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cos_timeseries(all_data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for e, data in all_data.items():
        cfg = EXPERIMENTS[e]
        s = data["encoder"]["steps"]
        ax.plot(s, smooth(data["encoder"]["cos"], 80),
                color=cfg["color"], alpha=0.85, linewidth=1.3, label=cfg["label"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training step"); ax.set_ylabel("cos (smoothed)")
    ax.set_title("Gradient cosine over training (encoder)")
    ax.set_ylim(-0.5, 0.5); ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "cos_timeseries.png"
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  Saved: {path}")


def plot_r_timeseries(all_data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for e, data in all_data.items():
        cfg = EXPERIMENTS[e]
        s = data["encoder"]["steps"]
        r = effective_r(data["encoder"]["d_over_t"], cfg["alpha"], cfg["beta"])
        ax.plot(s, smooth(r, 80), color=cfg["color"], alpha=0.85,
                linewidth=1.3, label=cfg["label"])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="r = 1")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"effective $r$ (smoothed, log)")
    ax.set_title(r"Effective gradient ratio $r$ over training (encoder)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "r_timeseries.png"
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures/gradient_norm")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading probe logs...")
    all_data = {}
    for e, cfg in EXPERIMENTS.items():
        if cfg["log"].exists():
            all_data[e] = parse_log(cfg["log"], cfg["alpha"], cfg["beta"])
            n = len(all_data[e]["encoder"]["steps"])
            print(f"  {e}: {n} probe points")
        else:
            print(f"  {e}: log not found ({cfg['log']})")
    if not all_data:
        return

    print("\nGenerating figures...")
    plot_bar_summary(all_data, out_dir)
    plot_cos_hist(all_data, out_dir)
    plot_cos_timeseries(all_data, out_dir)
    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
