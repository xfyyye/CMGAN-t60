"""Compare five main-result methods on the same 43 fixed-RIR samples."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cmgan_case_boxplot_mpl_20261028")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "outputs/real_rir_case_boxplots_2026-10-28_v1/plot_data.csv"
SELECTION = ROOT / "outputs/real_rir_case_candidates_2026-10-27_v1/metadata_selection_record.json"
REFERENCE = ROOT / "docs/noiseaware.pdf"
PROTECTED = [ROOT / f"docs/{name}" for name in ("main_zh.md", "main.tex", "main_bilingual.md")]
CASES = ("R13", "R47", "R61")
SCENES = {"R13": "AIR meeting", "R47": "AIR lecture", "R61": "AIR stairway"}
COUNTS = {"R13": {"test2": 7, "test4": 6}, "R47": {"test2": 9, "test4": 10}, "R61": {"test2": 6, "test4": 5}}
MODELS = (
    ("berp", "BERP", "#557DA5", "papercompare/berp_t60_v7_fixed/test_results"),
    ("noiseaware", "noise-aware", "#38958C", "papercompare/noiseaware/fix-v7dataset-test"),
    ("damtl", "DAMTL", "#8E74A6", "papercompare/damtl/test"),
    ("single_kan", "单任务 KAN", "#78818E", "runs/single_t60_kan_v2_mse_seed42/test"),
    ("ours", "Ours", "#D56838", "runs/kan_multitask_norm_w_denoise_b005_seed42/test"),
)
EXPECTED_RMSE = {
    "R13": (58.961, 36.874, 84.989, 45.883, 21.628),
    "R47": (93.608, 104.952, 83.785, 121.157, 64.160),
    "R61": (333.159, 294.186, 304.126, 244.026, 286.439),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_path(model: str, folder: str, split: str) -> Path:
    if model == "noiseaware":
        return ROOT / folder / f"fix_normalized_run1_test_{split}" / "detailed_results.csv"
    return ROOT / folder / f"evaluation_results_{split}.json"


def main() -> None:
    archives = {(model, split): archive_path(model, folder, split) for model, _, _, folder in MODELS for split in ("test2", "test4")}
    input_paths = [PREVIOUS, SELECTION, REFERENCE, *archives.values(), *PROTECTED]
    hashes_before = {str(p): digest(p) for p in input_paths}
    samples = [r for r in read_csv(PREVIOUS) if r["model"] == "kan_multi"]
    assert len(samples) == 43
    assert len({(r["split"], r["sample_name"]) for r in samples}) == 43
    assert set(r["case_id"] for r in samples) == set(CASES)
    samples.sort(key=lambda r: (CASES.index(r["case_id"]), r["split"], r["sample_name"]))
    selection = json.loads(SELECTION.read_text())
    meta = {r["rir_id"]: r for r in selection["rows"] if r["rir_id"] in CASES}
    for case in CASES:
        rs = [r for r in samples if r["case_id"] == case]
        assert dict(Counter(r["split"] for r in rs)) == COUNTS[case]
        assert all(r["content_sha256"] == meta[case]["sha256"] and r["original_filename"] == meta[case]["original_filename"] for r in rs)

    indexes = {}
    archive_metadata = []
    for (model, split), path in archives.items():
        if model == "noiseaware":
            raw_rows = read_csv(path)
            rows = [{"sample_name": r["Sample_Name"], "true_t60": r["T60_True"], "pred_t60": r["T60_Pred"], "snr": r["SNR_dB"]} for r in raw_rows]
            metadata = {"format": "detailed_results.csv", "source_row_count": len(rows)}
        else:
            data = json.loads(path.read_text())
            assert data["test_split"] == split
            rows = data["per_sample_results"]
            metadata = {k: data.get(k) for k in ("experiment_name", "dataset_version", "test_split")}
            metadata["source_row_count"] = len(rows)
        assert len(rows) == 1080
        index = {r["sample_name"]: r for r in rows}
        assert len(index) == len(rows), (path, "duplicate sample_name")
        indexes[model, split] = index
        archive_metadata.append({"model": model, "split": split, "path": str(path), **metadata})

    plot_rows = []
    values = {}
    metric_rows = []
    for case in CASES:
        case_samples = [r for r in samples if r["case_id"] == case]
        for model_index, (model, label, _, _) in enumerate(MODELS):
            errors, relative = [], []
            for sample in case_samples:
                row = indexes[model, sample["split"]][sample["sample_name"]]
                assert Decimal(str(row["true_t60"])) == Decimal(sample["historical_t60_s"])
                assert Decimal(str(row["snr"])) == Decimal(sample["snr_db"])
                reference_s, prediction_s = float(row["true_t60"]), float(row["pred_t60"])
                assert reference_s > 0 and math.isfinite(prediction_s)
                error_ms = 1000.0 * (prediction_s - reference_s)
                relative_pct = 100.0 * (prediction_s - reference_s) / reference_s
                if model == "ours":
                    assert abs(prediction_s - float(sample["prediction_t60_s"])) < 1e-12
                    assert abs(error_ms - float(sample["signed_error_ms"])) < 1e-9
                errors.append(error_ms)
                relative.append(relative_pct)
                plot_rows.append({
                    "case_id": case, "scene": SCENES[case], "split": sample["split"],
                    "sample_name": sample["sample_name"], "snr_db": sample["snr_db"],
                    "content_sha256": sample["content_sha256"], "original_filename": sample["original_filename"],
                    "historical_t60_s": reference_s, "model": model, "model_label": label,
                    "prediction_t60_s": prediction_s, "signed_error_ms": error_ms,
                    "signed_relative_error_pct": relative_pct, "prediction_archive_path": str(archives[model, sample["split"]]),
                })
            errors, relative = np.array(errors), np.array(relative)
            values[case, model] = errors, relative
            q1, median, q3 = np.percentile(errors, [25, 50, 75], method="linear")
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            assert abs(rmse - EXPECTED_RMSE[case][model_index]) < 0.0006
            metric_rows.append({
                "case_id": case, "model": model, "model_label": label, "n": len(errors),
                "historical_t60_s": meta[case]["historical_t60_s"], "rmse_ms": rmse,
                "mae_ms": float(np.mean(np.abs(errors))), "bias_ms": float(np.mean(errors)),
                "min_ms": float(np.min(errors)), "q1_ms": float(q1), "median_ms": float(median),
                "q3_ms": float(q3), "max_ms": float(np.max(errors)),
                "n_underestimated": int(np.sum(errors < 0)), "n_overestimated": int(np.sum(errors > 0)),
            })

    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
        "font.size": 11, "axes.unicode_minus": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.labelcolor": "#26323C", "text.color": "#26323C",
        "xtick.color": "#39434C", "ytick.color": "#39434C", "axes.edgecolor": "#AEB8C1",
        "axes.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 10.7))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.815, bottom=0.18, hspace=0.64)
    fig.suptitle("三条实测 RIR 案例：与基线方法的 T60 误差对比", fontsize=19, y=0.985)
    fig.text(0.5, 0.946, "Test2 与 Test4 合并，共 43 条语音，五种方法使用完全相同的样本", ha="center", fontsize=11.5, color="#5A6670")
    fig.legend(handles=[Patch(facecolor=color, alpha=0.60, edgecolor=color, label=label) for _, label, color, _ in MODELS],
               loc="upper center", bbox_to_anchor=(0.5, 0.926), ncol=5, frameon=False, fontsize=11.5, columnspacing=2.4, handlelength=1.7)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color="#64707B", markersize=5, label="Test2 样本"),
        Line2D([], [], marker="^", linestyle="none", color="#64707B", markersize=5, label="Test4 样本"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#26323C", markersize=5, label="均值"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.886), ncol=3, frameon=False, fontsize=10.5, columnspacing=2.8)

    centres = np.array([1.0, 2.65, 4.30])
    offsets = np.linspace(-0.44, 0.44, len(MODELS))
    tick_labels = [f"{c}  {SCENES[c]}\nT60 = {meta[c]['historical_t60_s']:.3f} s，n = {sum(COUNTS[c].values())}\nTest2 / Test4 = {COUNTS[c]['test2']} / {COUNTS[c]['test4']}" for c in CASES]
    plotted_counts = [0, 0]
    for panel_index, ax in enumerate(axes):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DCE2E7", linewidth=0.7, linestyle=(0, (2, 3)))
        ax.axhline(0, color="#59636E", linewidth=1.1, linestyle=(0, (5, 3)), zorder=1)
        for boundary in (1.825, 3.475):
            ax.axvline(boundary, color="#E8EBEE", linewidth=0.9, zorder=0)
        for case_index, case in enumerate(CASES):
            case_samples = [r for r in samples if r["case_id"] == case]
            jitter = np.random.default_rng(20261028 + case_index).permutation(np.linspace(-0.048, 0.048, len(case_samples)))
            splits = np.array([r["split"] for r in case_samples])
            for model_index, (model, _, color, _) in enumerate(MODELS):
                ys = values[case, model][panel_index]
                xpos = centres[case_index] + offsets[model_index]
                linewidth = 2.0 if model == "ours" else 1.5
                ax.boxplot([ys], positions=[xpos], widths=0.15, whis=(0, 100), showfliers=False,
                           notch=False, patch_artist=True, manage_ticks=False,
                           boxprops={"facecolor": color + "25", "edgecolor": color, "linewidth": linewidth},
                           medianprops={"color": "#26323C", "linewidth": 1.7},
                           whiskerprops={"color": color, "linewidth": 1.15}, capprops={"color": color, "linewidth": 1.3})
                for split, marker in (("test2", "o"), ("test4", "^")):
                    mask = splits == split
                    ax.scatter(xpos + jitter[mask], ys[mask], s=17 if marker == "o" else 21,
                               marker=marker, color=color, alpha=0.71, edgecolors="white", linewidths=0.3, zorder=3)
                    plotted_counts[panel_index] += int(np.sum(mask))
                ax.scatter([xpos], [float(np.mean(ys))], marker="D", s=27,
                           facecolors="white", edgecolors="#26323C", linewidths=1.0, zorder=5)
        ax.set_xlim(0.25, 5.05)
        ax.set_xticks(centres, tick_labels, fontsize=10.5, linespacing=1.5)
        ax.tick_params(axis="x", length=0, pad=11)
        ax.tick_params(axis="y", labelsize=10.5, length=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("(a) 有符号误差" if panel_index == 0 else "(b) 按各案例参考 T60 归一化的有符号误差", loc="left", fontsize=12, pad=13)
    axes[0].set_ylabel("估计误差 (ms)", fontsize=12, labelpad=10)
    axes[0].set_ylim(-665, 310)
    axes[0].set_yticks(np.arange(-600, 301, 150))
    axes[1].set_ylabel("相对估计误差 (%)", fontsize=12, labelpad=10)
    axes[1].set_ylim(-60, 95)
    axes[1].set_yticks(np.arange(-60, 91, 30))
    fig.text(0.085, 0.09, "箱体：第 25-75 百分位；横线：中位数；箱须：最小值至最大值。散点保留全部样本。", fontsize=10.3)
    fig.text(0.085, 0.062, "误差 = 预测值 - 历史参考标签；相对误差 = (预测值 - 参考标签) / 参考标签 × 100%。正值高估，负值低估。", fontsize=10.3)
    fig.text(0.085, 0.034, "单任务 KAN：论文主表的单任务 Fourier-KAN 基线；Ours：去噪辅助 KAN 多任务模型。", fontsize=10.3)
    assert plotted_counts == [215, 215]
    for panel_index, ax in enumerate(axes):
        all_values = np.concatenate([values[c, m][panel_index] for c in CASES for m, _, _, _ in MODELS])
        assert ax.get_ylim()[0] < float(np.min(all_values)) < float(np.max(all_values)) < ax.get_ylim()[1]
    png = OUT / "real_rir_cases_five_methods.png"
    pdf = OUT / "real_rir_cases_five_methods.pdf"
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Three measured-RIR cases: five-method T60 comparison"})
    plt.close(fig)
    write_csv(OUT / "plot_data.csv", plot_rows)
    write_csv(OUT / "summary_metrics.csv", metric_rows)
    assert all(digest(p) == hashes_before[str(p)] for p in input_paths)
    manifest = {
        "n_samples": len(samples), "n_predictions": len(plot_rows), "points_per_panel": plotted_counts,
        "cases": [{"case_id": c, "content_sha256": meta[c]["sha256"], "original_filename": meta[c]["original_filename"], "historical_t60_s": meta[c]["historical_t60_s"], "test_counts": COUNTS[c]} for c in CASES],
        "models": [{"key": m, "label": label, "color": color} for m, label, color, _ in MODELS],
        "archive_metadata": archive_metadata, "input_sha256": hashes_before, "inputs_and_manuscripts_unchanged": True,
        "source_row_counts": "Each of the 10 prediction archives has 1080 unique sample names.",
        "join": "exact (split,sample_name); reference T60 and SNR checked using Decimal",
        "all_selected_samples_joined": True, "ours_matches_previous_43_predictions": True,
        "error_ms": "1000 * (prediction_s - historical_reference_s)",
        "relative_error_pct": "100 * (prediction_s - historical_reference_s) / historical_reference_s",
        "quartiles": "numpy.percentile method=linear", "whiskers": "minimum and maximum", "diamond": "mean",
        "model_filtering": False, "sample_filtering": False, "snr_reweighting": False,
        "inference_training_or_audio_processing": False,
        "archive_precision_limit": "Values are recomputed from saved true/pred fields, often rounded to four decimals in seconds. No Error_ms field is used.",
        "dataset_version_note": "Existing archives retain their original dataset_version declarations, including v7 and v7_4s. This textual join does not newly establish historical waveform byte identity.",
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 三条实测 RIR：主结果方法对比\n\n"
        "沿用R13、R47、R61及全部43条语音（13/19/11条），不因预测效果更换案例或删样本。"
        "比较BERP、noise-aware、DAMTL、论文主表单任务Fourier-KAN和Ours，共215条已有预测。"
        "这里单任务不是参数量匹配MLP。\n\n"
        "十份本地存档各1080行且样本名唯一，按(split,sample_name)连接；所选43条的历史T60、SNR逐条一致，"
        "预测齐全有限。Ours的43条预测与上一版逐条一致。全部误差由预测减参考标签重算，不使用存档Error_ms。\n\n"
        "参考noiseaware §5.2图4，上排毫秒误差，下排相对误差百分比。箱须为最小值和最大值，箱体为25-75百分位，"
        "横线为中位数，菱形为均值。圆点/三角形分别表示Test2/Test4，所有样本均展示。"
        "无凹口、无显著性标记、无SNR重加权。\n\n"
        "plot_data.csv保存215行绘图数据，summary_metrics.csv保存15行分组指标，plot_manifest.json保存来源、"
        "存档版本及哈希、口径与连接核验。逐样本存档舍入精度及已有历史版本限制继续适用。\n\n"
        "这组图服务于主方法与对比基线的案例分析；参数量匹配MLP三模型图保留在上一独立输出目录，"
        "其比较目的属于消融分析。不同案例的语音/SNR/测量配置不同，不据此单独解释T60或模型机制。\n\n"
        "仅新增本目录的派生结果。没有修改论文、标签、模型、输入存档、旧图或共享代码，"
        "未进行模型推理、训练、音频处理及新数据合成。\n",
        encoding="utf-8",
    )
    print(json.dumps({"png": str(png), "pdf": str(pdf), "samples": len(samples), "predictions": len(plot_rows), "inputs_and_manuscripts_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
