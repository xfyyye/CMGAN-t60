"""Plot the three fixed measured-RIR cases from archived local predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = Path("/Users/xiexinni/case study/case_study_candidate_stats_2026-10-27_v1/sample_predictions_joined.csv")
SELECTION = ROOT / "outputs/real_rir_case_candidates_2026-10-27_v1/metadata_selection_record.json"
REFERENCE = ROOT / "docs/noiseaware.pdf"
PROTECTED = [ROOT / f"docs/{name}" for name in ("main_zh.md", "main.tex", "main_bilingual.md")]
CASES = ("R13", "R47", "R61")
SCENES = {"R13": "AIR meeting", "R47": "AIR lecture", "R61": "AIR stairway"}
MODELS = (
    ("matched_mlp_single", "匹配 MLP 单任务", "#4477AA"),
    ("matched_mlp_multi", "匹配 MLP 多任务", "#D58B23"),
    ("kan_multi", "KAN 多任务", "#228877"),
)
EXPECTED_COUNTS = {"R13": {"test2": 7, "test4": 6}, "R47": {"test2": 9, "test4": 10}, "R61": {"test2": 6, "test4": 5}}
EXPECTED_RMSE = {
    "R13": (37.92264393574007, 38.90139408219789, 21.628204520237706),
    "R47": (122.41216614802893, 83.26537474592268, 64.16017784461505),
    "R61": (237.27171229319657, 314.3320031142521, 286.4391732986255),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    input_paths = [SOURCE, SELECTION, REFERENCE, *PROTECTED]
    hashes_before = {str(path): digest(path) for path in input_paths}
    selection = json.loads(SELECTION.read_text())
    meta = {row["rir_id"]: row for row in selection["rows"] if row["rir_id"] in CASES}
    assert set(meta) == set(CASES)
    assert set(CASES).issubset(selection["primary_ids"])
    case_by_hash = {meta[case]["sha256"]: case for case in CASES}
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        source_rows = list(csv.DictReader(f))
    samples = [row for row in source_rows if row["content_sha256"] in case_by_hash]
    assert len(samples) == 43
    assert len({(row["split"], row["sample_name"]) for row in samples}) == 43
    samples.sort(key=lambda row: (CASES.index(case_by_hash[row["content_sha256"]]), row["split"], row["sample_name"]))
    values = {}
    long_rows = []
    for case in CASES:
        case_samples = [row for row in samples if case_by_hash[row["content_sha256"]] == case]
        assert dict(Counter(row["split"] for row in case_samples)) == EXPECTED_COUNTS[case]
        for model, _, _ in MODELS:
            error_ms = []
            relative_pct = []
            for row in case_samples:
                reference_s = float(row["historical_t60_s"])
                prediction_s = float(row[f"{model}_pred_t60_s"])
                assert row[f"{model}_join_status"] == "ok"
                assert row["original_filename"] == meta[case]["original_filename"]
                assert abs(reference_s - meta[case]["historical_t60_s"]) < 1e-12
                assert reference_s > 0 and math.isfinite(prediction_s)
                signed_ms = (prediction_s - reference_s) * 1000.0
                signed_pct = (prediction_s - reference_s) / reference_s * 100.0
                error_ms.append(signed_ms)
                relative_pct.append(signed_pct)
                long_rows.append({
                    "case_id": case, "scene": SCENES[case], "split": row["split"],
                    "sample_name": row["sample_name"], "snr_db": row["snr_db"],
                    "content_sha256": row["content_sha256"],
                    "original_filename": row["original_filename"],
                    "historical_t60_s": reference_s, "model": model,
                    "prediction_t60_s": prediction_s, "signed_error_ms": signed_ms,
                    "signed_relative_error_pct": signed_pct,
                    "prediction_archive_path": row[f"{model}_archive_path"],
                })
            values[case, model] = (np.array(error_ms), np.array(relative_pct))

    metric_rows = []
    for case in CASES:
        for model_index, (model, label, _) in enumerate(MODELS):
            errors, _ = values[case, model]
            q1, median, q3 = np.percentile(errors, [25, 50, 75], method="linear")
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            assert abs(rmse - EXPECTED_RMSE[case][model_index]) < 1e-9
            metric_rows.append({
                "case_id": case, "model": model, "model_label": label, "n": len(errors),
                "historical_t60_s": meta[case]["historical_t60_s"],
                "rmse_ms": rmse, "mae_ms": float(np.mean(np.abs(errors))),
                "bias_ms": float(np.mean(errors)), "min_ms": float(np.min(errors)),
                "q1_ms": float(q1), "median_ms": float(median), "q3_ms": float(q3),
                "max_ms": float(np.max(errors)),
            })

    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": font_name, "font.size": 11, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.labelcolor": "#26323C", "text.color": "#26323C",
        "xtick.color": "#39434C", "ytick.color": "#39434C",
        "axes.edgecolor": "#AEB8C1", "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 1, figsize=(11.6, 10.3))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.815, bottom=0.165, hspace=0.62)
    fig.suptitle("三条实测 RIR 的 T60 估计误差", fontsize=20, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, "Test2 与 Test4 合并，共 43 条语音，每个模型使用相同样本", ha="center", fontsize=11.5, color="#5A6670")
    fig.legend(handles=[Patch(facecolor=color, alpha=0.55, edgecolor=color, label=label) for _, label, color in MODELS],
               loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False, fontsize=11.5, handlelength=1.7, columnspacing=2.5)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color="#64707B", markersize=5, label="Test2 样本"),
        Line2D([], [], marker="^", linestyle="none", color="#64707B", markersize=5, label="Test4 样本"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#26323C", markersize=5, label="均值"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=3, frameon=False, fontsize=10.5, columnspacing=2.8)

    group_centres = np.array([1.0, 2.35, 3.70])
    model_offsets = [-0.27, 0.0, 0.27]
    displayed_points = [0, 0]
    labels = []
    for case in CASES:
        counts = EXPECTED_COUNTS[case]
        labels.append(f"{case}  {SCENES[case]}\nT60 = {meta[case]['historical_t60_s']:.3f} s，n = {sum(counts.values())}\nTest2 / Test4 = {counts['test2']} / {counts['test4']}")

    for panel_index, ax in enumerate(axes):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DCE2E7", linewidth=0.7, linestyle=(0, (2, 3)))
        ax.axhline(0, color="#59636E", linewidth=1.1, linestyle=(0, (5, 3)), zorder=1)
        for boundary in (1.675, 3.025):
            ax.axvline(boundary, color="#E8EBEE", linewidth=0.9, zorder=0)
        for case_index, case in enumerate(CASES):
            case_samples = [row for row in samples if case_by_hash[row["content_sha256"]] == case]
            # Fixed, cosmetic jitter only. Same sample order and offsets for all models/panels.
            jitter = np.random.default_rng(20261028 + case_index).permutation(np.linspace(-0.065, 0.065, len(case_samples)))
            splits = np.array([row["split"] for row in case_samples])
            for model_index, (model, _, color) in enumerate(MODELS):
                ys = values[case, model][panel_index]
                xpos = group_centres[case_index] + model_offsets[model_index]
                ax.boxplot([ys], positions=[xpos], widths=0.19, whis=(0, 100),
                           showfliers=False, notch=False, patch_artist=True, manage_ticks=False,
                           boxprops={"facecolor": color + "25", "edgecolor": color, "linewidth": 1.65},
                           medianprops={"color": "#26323C", "linewidth": 1.8},
                           whiskerprops={"color": color, "linewidth": 1.2},
                           capprops={"color": color, "linewidth": 1.4})
                for split, marker in (("test2", "o"), ("test4", "^")):
                    mask = splits == split
                    ax.scatter(xpos + jitter[mask], ys[mask], s=21 if marker == "o" else 25,
                               marker=marker, color=color, alpha=0.69, edgecolors="white", linewidths=0.35, zorder=3)
                    displayed_points[panel_index] += int(np.sum(mask))
                ax.scatter([xpos], [float(np.mean(ys))], marker="D", s=29,
                           facecolors="white", edgecolors="#26323C", linewidths=1.0, zorder=5)
        ax.set_xlim(0.42, 4.28)
        ax.set_xticks(group_centres, labels, fontsize=10.5, linespacing=1.55)
        ax.tick_params(axis="x", length=0, pad=11)
        ax.tick_params(axis="y", labelsize=10.5, length=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("(a) 有符号误差" if panel_index == 0 else "(b) 按各案例参考 T60 归一化的有符号误差", loc="left", fontsize=12, pad=13)
    axes[0].set_ylabel("估计误差 (ms)", fontsize=12, labelpad=10)
    axes[0].set_ylim(-665, 290)
    axes[0].set_yticks(np.arange(-600, 201, 200))
    axes[1].set_ylabel("相对估计误差 (%)", fontsize=12, labelpad=10)
    axes[1].set_ylim(-60, 40)
    axes[1].set_yticks(np.arange(-60, 41, 20))
    fig.text(0.10, 0.072, "箱体：第 25-75 百分位；横线：中位数；箱须：最小值至最大值。散点保留全部样本。", fontsize=10.3)
    fig.text(0.10, 0.043, "误差 = 预测值 - 历史参考标签；相对误差 = 误差 / 参考标签 × 100%。正值为高估，负值为低估。", fontsize=10.3)

    assert displayed_points == [129, 129]
    for panel_index, ax in enumerate(axes):
        all_values = np.concatenate([values[case, model][panel_index] for case in CASES for model, _, _ in MODELS])
        assert ax.get_ylim()[0] < float(np.min(all_values)) < float(np.max(all_values)) < ax.get_ylim()[1]
    png_path = OUT / "real_rir_three_cases_boxplots.png"
    pdf_path = OUT / "real_rir_three_cases_boxplots.pdf"
    fig.savefig(png_path, dpi=220, facecolor="white")
    fig.savefig(pdf_path, facecolor="white", metadata={"Title": "Three measured-RIR cases: signed T60 estimation errors", "Author": "CMGAN-t60 Case Study"})
    plt.close(fig)
    write_csv(OUT / "plot_data.csv", long_rows)
    write_csv(OUT / "summary_metrics.csv", metric_rows)

    hashes_after = {str(path): digest(path) for path in input_paths}
    assert hashes_before == hashes_after, "An input or protected manuscript changed."
    manifest = {
        "selected_cases": [{"case_id": case, "scene": SCENES[case], "metadata": meta[case], "test_sample_counts": EXPECTED_COUNTS[case]} for case in CASES],
        "n_speech_samples": len(samples), "n_predictions": len(long_rows),
        "scatter_points_per_panel": displayed_points,
        "model_definitions": {model: label for model, label, _ in MODELS},
        "input_sha256_before": hashes_before, "inputs_and_manuscripts_unchanged": hashes_before == hashes_after,
        "error_ms": "1000 * (predicted_t60_s - historical_t60_s)",
        "relative_error_pct": "100 * (predicted_t60_s - historical_t60_s) / historical_t60_s",
        "quartiles": "numpy.percentile method=linear", "whiskers": "minimum and maximum, whis=(0,100)",
        "box_centre_line": "median", "diamond": "arithmetic mean", "notches": False,
        "all_samples_retained": True, "snr_reweighted": False,
        "jitter_seed": 20261028, "jitter_is_horizontal_only": True,
        "inference_training_or_audio_processing": False,
        "reference_layout": "noiseaware.pdf section 5.2, Fig.4, PDF page 6",
        "matplotlib_version": matplotlib.__version__,
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 三条实测 RIR 案例误差箱线图\n\n"
        "使用此前按元数据固定的 R13、R47、R61，合并其 Test2 与 Test4 的全部 43 条语音。"
        "三模型各 43 条，共 129 条存档预测，未筛掉任何样本，未按 SNR 重加权。\n\n"
        "上图为有符号毫秒误差，下图为除以各案例历史参考 T60 的有符号百分比误差。"
        "统一采用预测减历史参考标签，正值高估、负值低估。参考 noiseaware §5.2 图4，箱须为最小值和最大值，"
        "箱体为第25至75百分位，横线为中位数。叠加全部样本点（圆形Test2、三角Test4）及均值菱形。"
        "不使用箱体凹口或显著性标记。每个案例的参考标签固定，因此相对误差面板是同一分布的比例缩放。\n\n"
        "R13 AIR meeting：0.303s，13条（Test2=7、Test4=6）。\n\n"
        "R47 AIR lecture：0.697s，19条（Test2=9、Test4=10）。\n\n"
        "R61 AIR stairway：1.133s，11条（Test2=6、Test4=5）。\n\n"
        "三个模型为 dropout0.1 参数量匹配MLP单任务、dropout0.1 参数量匹配MLP多任务及KAN多任务，"
        "多任务均b005、seed42。预测沿用已有存档，保存精度的限制仍然适用。跨案例的语音、SNR和测量配置不同，"
        "本图描述固定RIR案例，不把跨案例差异单独归因于T60或宣称显著性。\n\n"
        "plot_data.csv 为129行逐模型绘图数据；summary_metrics.csv 为9行指标及五数概括；"
        "plot_manifest.json 保存来源路径、输入哈希、选择及绘图口径。plot_cases.py 为可重现脚本。"
        "仅新增本目录产物，未修改论文、输入CSV、历史标签、配置、模型或已有结果；无推理、训练及音频处理。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_png": str(png_path), "output_pdf": str(pdf_path), "samples": len(samples), "predictions": len(long_rows), "points_per_panel": displayed_points, "inputs_and_manuscripts_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
