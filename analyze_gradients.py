"""
梯度探针可视化分析脚本
Usage:
    python analyze_gradients.py [--out_dir figures/gradient]

生成图表:
    1. cos_timeseries.png   — 四组实验 encoder cos 随训练步数折线图
    2. r_timeseries.png     — 四组实验 encoder r 随训练步数折线图
    3. cos_r_bar.png        — 四组实验 cos / r / neg% 均值对比柱状图（含 T60 性能叠加）
    4. cos_hist.png         — 四组实验 cos 分布直方图（区分正交型 vs 抵消型）
    5. layer_cos_r.png      — 三层（encoder/tscb_1/tscb_2）cos & r 均值对比（w_denoise vs w_balanced）
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent

EXPERIMENTS = {
    "w_denoise": {
        "log": ROOT / "runs/kan_multitask_w_denoise/nohup_kan_w_denoise.log",
        "label": r"w_denoise ($\alpha$=1.0, $\beta$=0.1)",
        "color": "#2196F3",
        "t60_rmse": 96.41,
    },
    "w_balanced": {
        "log": ROOT / "runs/kan_multitask_w_balanced/nohup_train.log",
        "label": r"w_balanced ($\alpha$=0.5, $\beta$=0.5)",
        "color": "#4CAF50",
        "t60_rmse": 100.28,
    },
    "w_eq": {
        "log": ROOT / "runs/kan_multitask_w_eq/nohup_kan_w_eq.log",
        "label": r"w_eq ($\alpha$=1.0, $\beta$=1.0)",
        "color": "#FF9800",
        "t60_rmse": 105.64,
    },
    "w_t60": {
        "log": ROOT / "runs/kan_multitask_w_t60/nohup_kan_w_t60.log",
        "label": r"w_t60 ($\alpha$=0.1, $\beta$=1.0)",
        "color": "#F44336",
        "t60_rmse": 107.84,
    },
}

LAYERS = ["encoder", "tscb_1", "tscb_2"]
PROBE_RE = re.compile(
    r"\[probe step=(\d+)\]"
    r".*?encoder:cos=([+-]?\d+\.\d+),r=(\d+\.\d+)"
    r".*?tscb_1:cos=([+-]?\d+\.\d+),r=(\d+\.\d+)"
    r".*?tscb_2:cos=([+-]?\d+\.\d+),r=(\d+\.\d+)"
)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ── 解析 log ─────────────────────────────────────────────────────────────────

def parse_log(log_path: Path):
    """返回 dict: {layer: {'steps': [...], 'cos': [...], 'r': [...]}}"""
    data = {l: {"steps": [], "cos": [], "r": []} for l in LAYERS}
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = PROBE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            vals = [
                (float(m.group(2)), float(m.group(3))),   # encoder
                (float(m.group(4)), float(m.group(5))),   # tscb_1
                (float(m.group(6)), float(m.group(7))),   # tscb_2
            ]
            for i, layer in enumerate(LAYERS):
                data[layer]["steps"].append(step)
                data[layer]["cos"].append(vals[i][0])
                data[layer]["r"].append(vals[i][1])
    for layer in LAYERS:
        for k in data[layer]:
            data[layer][k] = np.array(data[layer][k])
    return data


def load_all():
    all_data = {}
    for exp, cfg in EXPERIMENTS.items():
        if cfg["log"].exists():
            all_data[exp] = parse_log(cfg["log"])
            n = len(all_data[exp]["encoder"]["steps"])
            print(f"  {exp}: {n} probe points")
        else:
            print(f"  {exp}: log not found ({cfg['log']})")
    return all_data


def smooth(arr, window=50):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# ── 图1: cos 时间序列 ────────────────────────────────────────────────────────

def plot_cos_timeseries(all_data, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for exp, data in all_data.items():
        cfg = EXPERIMENTS[exp]
        steps = data["encoder"]["steps"]
        cos = smooth(data["encoder"]["cos"], window=100)
        ax.plot(steps, cos, color=cfg["color"], alpha=0.85, linewidth=1.3,
                label=cfg["label"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(ax.get_xlim(), -1, 0, alpha=0.04, color="red")
    ax.fill_between(ax.get_xlim(), 0, 1, alpha=0.04, color="green")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cosine Similarity (cos)")
    ax.set_title("Gradient Direction Alignment over Training (Encoder Layer, smoothed)")
    ax.set_ylim(-0.6, 0.6)
    ax.legend(loc="upper right")
    ax.text(ax.get_xlim()[1] * 0.01, -0.55, "conflict zone (cos < 0)",
            color="red", alpha=0.6, fontsize=8)
    ax.text(ax.get_xlim()[1] * 0.01, 0.48, "cooperative zone (cos > 0)",
            color="green", alpha=0.6, fontsize=8)
    fig.tight_layout()
    path = out_dir / "cos_timeseries.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 图2: r 时间序列 ──────────────────────────────────────────────────────────

def plot_r_timeseries(all_data, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for exp, data in all_data.items():
        cfg = EXPERIMENTS[exp]
        steps = data["encoder"]["steps"]
        r = smooth(data["encoder"]["r"], window=100)
        ax.plot(steps, r, color=cfg["color"], alpha=0.85, linewidth=1.3,
                label=cfg["label"])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="r = 1 (equal magnitude)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"Gradient Magnitude Ratio $r = \|g_{T60}\| / \|g_{denoise}\|$")
    ax.set_title("Gradient Magnitude Ratio over Training (Encoder Layer, smoothed)")
    ax.set_ylim(0, 3.5)
    ax.legend(loc="upper right")
    ax.text(ax.get_xlim()[1] * 0.6, 1.08, "T60 dominates (r > 1)",
            color="gray", fontsize=8)
    ax.text(ax.get_xlim()[1] * 0.6, 0.82, "denoise dominates (r < 1)",
            color="gray", fontsize=8)
    fig.tight_layout()
    path = out_dir / "r_timeseries.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 图3: 均值柱状图 + T60 RMSE 叠加 ─────────────────────────────────────────

def plot_bar_summary(all_data, out_dir: Path):
    exps = [e for e in EXPERIMENTS if e in all_data]
    labels = [EXPERIMENTS[e]["label"] for e in exps]
    colors = [EXPERIMENTS[e]["color"] for e in exps]
    rmses = [EXPERIMENTS[e]["t60_rmse"] for e in exps]

    cos_means = [np.mean(all_data[e]["encoder"]["cos"]) for e in exps]
    r_means   = [np.mean(all_data[e]["encoder"]["r"])   for e in exps]
    neg_pcts  = [np.mean(all_data[e]["encoder"]["cos"] < 0) * 100 for e in exps]

    x = np.arange(len(exps))
    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 子图A: cos 均值 ──
    ax_cos = fig.add_subplot(gs[0, 0])
    bars = ax_cos.bar(x, cos_means, color=colors, edgecolor="white", linewidth=0.5)
    ax_cos.axhline(0, color="black", linewidth=0.8)
    ax_cos.set_xticks(x); ax_cos.set_xticklabels(
        [e.replace("w_", "") for e in exps], fontsize=9)
    ax_cos.set_ylabel("Mean cos")
    ax_cos.set_title("A. Mean Gradient Cosine (Encoder)")
    ax_cos.set_ylim(-0.05, 0.06)
    for bar, v in zip(bars, cos_means):
        ax_cos.text(bar.get_x() + bar.get_width()/2, v + 0.001 if v >= 0 else v - 0.003,
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    # ── 子图B: r 均值 ──
    ax_r = fig.add_subplot(gs[0, 1])
    bars = ax_r.bar(x, r_means, color=colors, edgecolor="white", linewidth=0.5)
    ax_r.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_r.set_xticks(x); ax_r.set_xticklabels(
        [e.replace("w_", "") for e in exps], fontsize=9)
    ax_r.set_ylabel(r"Mean $r = \|g_{T60}\|/\|g_{denoise}\|$")
    ax_r.set_title("B. Mean Gradient Magnitude Ratio (Encoder)")
    for bar, v in zip(bars, r_means):
        ax_r.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── 子图C: neg% ──
    ax_neg = fig.add_subplot(gs[1, 0])
    bars = ax_neg.bar(x, neg_pcts, color=colors, edgecolor="white", linewidth=0.5)
    ax_neg.axhline(50, color="red", linewidth=0.8, linestyle="--", alpha=0.6,
                   label="50% (random)")
    ax_neg.set_xticks(x); ax_neg.set_xticklabels(
        [e.replace("w_", "") for e in exps], fontsize=9)
    ax_neg.set_ylabel("neg% (cos < 0)")
    ax_neg.set_title("C. Fraction of Conflicting Steps (Encoder)")
    ax_neg.set_ylim(0, 65)
    ax_neg.legend(fontsize=8)
    for bar, v in zip(bars, neg_pcts):
        ax_neg.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    # ── 子图D: r vs T60 RMSE 散点 ──
    ax_sc = fig.add_subplot(gs[1, 1])
    for exp, r, rmse, color in zip(exps, r_means, rmses, colors):
        ax_sc.scatter(r, rmse, color=color, s=120, zorder=5)
        ax_sc.annotate(exp.replace("w_", ""), (r, rmse),
                       xytext=(5, 4), textcoords="offset points", fontsize=8)
    # 趋势线
    z = np.polyfit(r_means, rmses, 1)
    p = np.poly1d(z)
    r_range = np.linspace(min(r_means)-0.05, max(r_means)+0.1, 100)
    ax_sc.plot(r_range, p(r_range), "k--", linewidth=1, alpha=0.5)
    ax_sc.set_xlabel(r"Mean $r$ (T60/denoise gradient ratio)")
    ax_sc.set_ylabel("T60 RMSE (ms)")
    ax_sc.set_title("D. Gradient Ratio r vs T60 Performance")
    corr = np.corrcoef(r_means, rmses)[0, 1]
    ax_sc.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax_sc.transAxes,
               fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    fig.suptitle("Gradient Probe Analysis: Multi-task Denoise + T60 Estimation",
                 fontsize=13, fontweight="bold", y=1.01)
    path = out_dir / "cos_r_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 图4: cos 分布直方图 ─────────────────────────────────────────────────────

def plot_cos_hist(all_data, out_dir: Path):
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (exp, data) in zip(axes, all_data.items()):
        cfg = EXPERIMENTS[exp]
        cos_vals = data["encoder"]["cos"]
        ax.hist(cos_vals, bins=60, color=cfg["color"], alpha=0.8, edgecolor="white",
                linewidth=0.3, density=True)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        mean_val = np.mean(cos_vals)
        ax.axvline(mean_val, color="darkred", linewidth=1.5, linestyle="-",
                   label=f"mean={mean_val:+.3f}")
        neg_pct = np.mean(cos_vals < 0) * 100
        ax.set_title(f"{exp}\nneg%={neg_pct:.1f}%", fontsize=10)
        ax.set_xlabel("cos similarity")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=8)
        # 标注分布类型
        std_val = np.std(cos_vals)
        if abs(mean_val) < 0.05 and std_val < 0.25:
            tag = "orthogonal\n(cos≈0, low std)"
        elif abs(mean_val) < 0.05 and std_val >= 0.25:
            tag = "cancellation\n(cos≈0, high std)"
        elif mean_val > 0.1:
            tag = "cooperative\n(cos>0)"
        else:
            tag = "conflicting\n(cos<0)"
        ax.text(0.05, 0.90, tag, transform=ax.transAxes, fontsize=8,
                va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))
    axes[0].set_ylabel("Density")
    fig.suptitle("Gradient Cosine Distribution per Experiment (Encoder Layer)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "cos_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 图5: 分层 cos & r 均值对比 (w_denoise vs w_balanced) ────────────────────

def plot_layer_comparison(all_data, out_dir: Path):
    target_exps = [e for e in ["w_denoise", "w_balanced", "w_eq", "w_t60"]
                   if e in all_data]
    layers_display = ["Encoder", "TSCB-1", "TSCB-2"]
    x = np.arange(len(LAYERS))
    width = 0.8 / len(target_exps)

    fig, (ax_cos, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, exp in enumerate(target_exps):
        cfg = EXPERIMENTS[exp]
        data = all_data[exp]
        cos_means = [np.mean(data[l]["cos"]) for l in LAYERS]
        r_means   = [np.mean(data[l]["r"])   for l in LAYERS]
        offset = (i - len(target_exps)/2 + 0.5) * width
        ax_cos.bar(x + offset, cos_means, width, color=cfg["color"],
                   label=exp, edgecolor="white", linewidth=0.5)
        ax_r.bar(x + offset, r_means, width, color=cfg["color"],
                 label=exp, edgecolor="white", linewidth=0.5)

    ax_cos.axhline(0, color="black", linewidth=0.8)
    ax_cos.set_xticks(x); ax_cos.set_xticklabels(layers_display)
    ax_cos.set_ylabel("Mean cos")
    ax_cos.set_title("Gradient Cosine by Layer")
    ax_cos.legend(fontsize=8)

    ax_r.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_r.set_xticks(x); ax_r.set_xticklabels(layers_display)
    ax_r.set_ylabel(r"Mean $r$")
    ax_r.set_title("Gradient Magnitude Ratio by Layer")
    ax_r.legend(fontsize=8)

    fig.suptitle("Per-layer Gradient Analysis across Experiments",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "layer_cos_r.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="figures/gradient")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading probe logs...")
    all_data = load_all()
    if not all_data:
        print("No data found.")
        return

    print("\nGenerating figures...")
    plot_cos_timeseries(all_data, out_dir)
    plot_r_timeseries(all_data, out_dir)
    plot_bar_summary(all_data, out_dir)
    plot_cos_hist(all_data, out_dir)
    plot_layer_comparison(all_data, out_dir)
    print(f"\nDone. All figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
