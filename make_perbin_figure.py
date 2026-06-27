"""Generate the per-bin analysis figure (Fig. 6 in the paper).

Averages bin_analysis / snr_analysis over test1--test4 for the proposed
(norm_w_denoise) model and the single-task baseline, then plots:
  (a) RMSE vs T60 bin
  (b) RMSE vs SNR bin
with the two models overlaid.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
PROPOSED = ROOT / "runs/kan_multitask_norm_w_denoise/test"
SINGLE   = ROOT / "runs/single_t60_kan_v2_mae_clip1_log/test"
OUT      = ROOT / "docs/figs"

TESTS = ["test1", "test2", "test3", "test4"]


def aggregate(runs_dir, field, metric="rmse_ms"):
    """Average `metric` per bin across the 4 test sets. Returns (labels, vals)."""
    agg = {}
    for sp in TESTS:
        d = json.load(open(runs_dir / f"evaluation_results_{sp}.json"))
        for binname, stats in d[field].items():
            agg.setdefault(binname, []).append(stats[metric])
    labels = list(agg.keys())
    vals = [float(np.mean(agg[l])) for l in labels]
    return labels, vals


def short_t60(label):
    # 'T60_0.1-0.3s' -> '0.1-0.3'
    return label.replace("T60_", "").replace("s", "")

def short_snr(label):
    # 'SNR_-5dB' -> '-5'
    return label.replace("SNR_", "").replace("dB", "")


def main():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.edgecolor": "#333",
    })
    BLUE = "#2196F3"; GREY = "#9E9E9E"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) RMSE vs T60 bin
    lab_p, val_p = aggregate(PROPOSED, "bin_analysis")
    lab_s, val_s = aggregate(SINGLE,   "bin_analysis")
    x = np.arange(len(lab_p))
    w = 0.38
    ax1.bar(x - w/2, val_s, w, color=GREY, edgecolor="white", linewidth=0.5,
            label="Single-task")
    ax1.bar(x + w/2, val_p, w, color=BLUE, edgecolor="white", linewidth=0.5,
            label="Proposed (denoise-dom.)")
    ax1.set_xticks(x); ax1.set_xticklabels([short_t60(l) for l in lab_p], fontsize=9)
    ax1.set_xlabel(r"$T_{60}$ bin (s)")
    ax1.set_ylabel("RMSE (ms)")
    ax1.set_title(r"(a) RMSE vs. $T_{60}$ bin")
    ax1.legend(fontsize=9)

    # (b) RMSE vs SNR bin
    lab_p, val_p = aggregate(PROPOSED, "snr_analysis")
    lab_s, val_s = aggregate(SINGLE,   "snr_analysis")
    x = np.arange(len(lab_p))
    ax2.plot(x, val_s, "-o", color=GREY, linewidth=1.6, markersize=6, label="Single-task")
    ax2.plot(x, val_p, "-o", color=BLUE, linewidth=1.6, markersize=6, label="Proposed (denoise-dom.)")
    ax2.set_xticks(x); ax2.set_xticklabels([short_snr(l) for l in lab_p], fontsize=9)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("RMSE (ms)")
    ax2.set_title(r"(b) RMSE vs. SNR bin")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_bin.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")

    # Print table for the paper text
    print("\n=== T60-bin RMSE (proposed / single) ===")
    lp,vp = aggregate(PROPOSED,"bin_analysis"); ls,vs = aggregate(SINGLE,"bin_analysis")
    for l,p,s in zip(lp,vp,vs):
        print(f"  {short_t60(l):>8s}  prop={p:6.2f}  single={s:6.2f}  diff={s-p:+.2f}")
    print("=== SNR-bin RMSE ===")
    lp,vp = aggregate(PROPOSED,"snr_analysis"); ls,vs = aggregate(SINGLE,"snr_analysis")
    for l,p,s in zip(lp,vp,vs):
        print(f"  {short_snr(l):>5s}dB  prop={p:6.2f}  single={s:6.2f}  diff={s-p:+.2f}")


if __name__ == "__main__":
    main()
