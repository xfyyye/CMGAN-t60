"""Minimal four-method case boxplots using the already verified predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cmgan_case_boxplot_mpl_20261028")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "outputs/real_rir_case_baselines_2026-10-28_v1/plot_data.csv"
SOURCE_METRICS = SOURCE.parent / "summary_metrics.csv"
PROTECTED = [ROOT / f"docs/{name}" for name in ("main_zh.md", "main.tex", "main_bilingual.md")]
CASES = ("R13", "R47", "R61")
MODELS = (
    ("berp", "BERP", "#63AA69"),
    ("noiseaware", "noise-aware", "#4D8CC2"),
    ("damtl", "DAMTL", "#F0B27B"),
    ("ours", "Ours", "#C43C27"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    before = {str(p): sha256(p) for p in [SOURCE, SOURCE_METRICS, *PROTECTED]}
    selected_models = {model for model, _, _ in MODELS}
    rows = [r for r in read_csv(SOURCE) if r["model"] in selected_models]
    assert len(rows) == 172
    assert len({(r["split"], r["sample_name"]) for r in rows}) == 43
    assert len({(r["split"], r["sample_name"], r["model"]) for r in rows}) == 172
    assert Counter(r["model"] for r in rows) == {m: 43 for m in selected_models}
    expected = {"R13": {"test2": 7, "test4": 6}, "R47": {"test2": 9, "test4": 10}, "R61": {"test2": 6, "test4": 5}}
    metrics = {(r["case_id"], r["model"]): r for r in read_csv(SOURCE_METRICS)}
    values = {}
    for case in CASES:
        for model, _, _ in MODELS:
            rs = [r for r in rows if r["case_id"] == case and r["model"] == model]
            assert dict(Counter(r["split"] for r in rs)) == expected[case]
            ms = np.array([1000 * (float(r["prediction_t60_s"]) - float(r["historical_t60_s"])) for r in rs])
            pct = np.array([100 * (float(r["prediction_t60_s"]) - float(r["historical_t60_s"])) / float(r["historical_t60_s"]) for r in rs])
            assert np.all(np.isfinite(ms)) and np.all(np.isfinite(pct))
            assert np.allclose(ms, [float(r["signed_error_ms"]) for r in rs], atol=1e-10, rtol=0)
            assert np.allclose(pct, [float(r["signed_relative_error_pct"]) for r in rs], atol=1e-10, rtol=0)
            assert abs(float(np.sqrt(np.mean(ms ** 2))) - float(metrics[case, model]["rmse_ms"])) < 1e-10
            values[case, model] = (ms, pct)

    font = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(font)).get_name(),
        "font.size": 11.5, "axes.labelsize": 12.5, "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5, "axes.unicode_minus": True, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "axes.edgecolor": "#333333",
        "axes.linewidth": 0.85, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 8.8))
    fig.subplots_adjust(left=0.145, right=0.98, top=0.985, bottom=0.12, hspace=0.48)
    centres = np.array([1.0, 2.0, 3.0])
    offsets = np.array([-0.24, -0.08, 0.08, 0.24])
    for panel, ax in enumerate(axes):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#BDBDBD", linestyle=":", linewidth=0.75)
        for model_index, (model, _, color) in enumerate(MODELS):
            linewidth = 2.0 if model == "ours" else 1.05
            ax.boxplot([values[case, model][panel] for case in CASES],
                       positions=centres + offsets[model_index], widths=0.115,
                       whis=(0, 100), showfliers=False, notch=False, patch_artist=True,
                       manage_ticks=False,
                       boxprops={"facecolor": "white", "edgecolor": color, "linewidth": linewidth},
                       medianprops={"color": color, "linewidth": linewidth},
                       whiskerprops={"color": color, "linewidth": linewidth},
                       capprops={"color": color, "linewidth": linewidth})
        ax.set_xlim(0.55, 3.45)
        ax.set_xticks(centres, CASES)
        ax.set_xlabel("RIR case", labelpad=3)
        ax.tick_params(direction="out", length=3, width=0.8, pad=3)
        ax.set_ylabel("Estimation error (ms)" if panel == 0 else "Estimation error (%)", labelpad=7)
        ax.text(0.5, -0.265, "(a)" if panel == 0 else "(b)",
                transform=ax.transAxes, ha="center", va="top", fontsize=14)
        legend = ax.legend(handles=[Line2D([], [], color=color, lw=2 if model == "ours" else 1.1, label=label)
                                    for model, label, color in MODELS],
                           loc="lower left" if panel == 0 else "upper right",
                           frameon=True, fancybox=False, framealpha=1, edgecolor="#D1D1D1",
                           fontsize=11.5, handlelength=2.2, labelspacing=0.3,
                           borderpad=0.35, handletextpad=0.65)
        legend.get_frame().set_linewidth(0.6)
    axes[0].set_ylim(-660, 320)
    axes[0].set_yticks([-600, -400, -200, 0, 200])
    axes[1].set_ylim(-60, 90)
    axes[1].set_yticks(np.arange(-60, 81, 20))
    for panel, ax in enumerate(axes):
        all_values = np.concatenate([values[c, m][panel] for c in CASES for m, _, _ in MODELS])
        assert ax.get_ylim()[0] < float(np.min(all_values)) < float(np.max(all_values)) < ax.get_ylim()[1]
    png, pdf = OUT / "real_rir_cases_four_methods.png", OUT / "real_rir_cases_four_methods.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Three measured-RIR cases: four-method estimation errors"})
    plt.close(fig)
    assert all(sha256(Path(p)) == value for p, value in before.items())
    manifest = {
        "source": str(SOURCE), "source_sha256": before[str(SOURCE)],
        "n_samples": 43, "n_predictions": 172,
        "cases": {case: {"historical_t60_s": float(next(r["historical_t60_s"] for r in rows if r["case_id"] == case)), "test_counts": expected[case]} for case in CASES},
        "models": [{"key": m, "label": label, "color": color} for m, label, color in MODELS],
        "removed_model": "single_kan", "all_four_method_samples_retained": True,
        "quartiles": "numpy linear interpolation", "whiskers": "minimum and maximum",
        "centre_line": "median", "scatter": False, "means": False, "notches": False,
        "error_ms": "1000 * (prediction_s - historical_reference_s)",
        "relative_error_pct": "100 * (prediction_s - historical_reference_s) / historical_reference_s",
        "inputs_and_manuscripts_unchanged": True, "input_sha256": before,
        "source_predictions_reused": True, "new_inference_or_training": False,
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 简洁版四方法箱线图\n\n"
        "仅比较BERP、noise-aware、DAMTL、Ours，去除单任务Fourier-KAN。"
        "沿用上一版已核验的43条语音与这四种方法的172条已有预测，未删除任何所选方法的样本。\n\n"
        "R13：AIR meeting，历史T60=0.303s，n=13；R47：AIR lecture，0.697s，n=19；"
        "R61：AIR stairway，1.133s，n=11。三个案例均合并Test2/Test4，分布分别为7/6、9/10、6/5。\n\n"
        "按用户参考图简化：上下两图分别为毫秒和相对百分比有符号误差，统一预测减参考。"
        "箱体为25-75百分位，中线为中位数，须为最小值/最大值。空心箱体与彩色线条，"
        "不显示散点、均值、凹口或统计显著性标记。保留全部极值，不改变数据或重新选择案例。\n\n"
        "数据源为上一版plot_data.csv，统计值已与其summary_metrics.csv逐组一致性核对。"
        "四方法仅作描述性比较，不提供训练分布、噪声或上下文机制的因果结论。"
        "输入、历史图与论文均未修改，无训练、推理、合成或标签重算。\n",
        encoding="utf-8",
    )
    print(json.dumps({"png": str(png), "pdf": str(pdf), "samples": 43, "predictions": 172, "inputs_and_manuscripts_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
