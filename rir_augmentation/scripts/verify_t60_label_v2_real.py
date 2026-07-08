"""
T60 物理增广 —— 标签精度验证脚本（文档 3，已按真实数据修正）

【背景】
初版脚本假设 RIR 是干净的、动态范围达 60dB。实测真实 RIR（OpenSLR28/ACE）
存在两个问题，已在本版修正：
  1. 直达声不在开头（peak 在 100~140ms 处），必须先截断
  2. 有效衰减动态常不足 60dB（40dB 左右就触噪声底），
     直接按 -60dB 区间回归会被噪声尾带飞
因此本版同时报告 T10/T20/T30 三种区间外推 T60 的结果，对照文件名标注。

【守门规则】（以 T20 外推 T60 为准）
  中位误差 < 10%   → 标签重算可靠，方案成立
  中位误差 10-25%  → 可用但需在论文中声明误差，或改用其他标签策略
  中位误差 > 25%   → Schroeder 重算不可靠，方案需调整（如保留原标签）

【用法】
    python verify_t60_label.py --rir_dir <真实RIR目录> --out report.csv

【依赖】
    pip install numpy soundfile
"""

import os
import re
import argparse
import numpy as np
import soundfile as sf


# ─── 核心 1：带预处理的 Schroeder T60 估计 ──────────────────────────

def schroeder_t60(h, sr, decay_db=20, onset_align=True):
    """从 RIR 估 T60（Schroeder 后向积分 + 直达声截断 + 区间外推）

    参数
    ----
    h : np.ndarray, (N,)
        RIR 波形（多声道请预先取单声道）
    sr : int
        采样率
    decay_db : int
        回归区间的衰减深度（10/20/30）。注意：真实 RIR 动态常不足 60dB，
        用 -60 区间会被噪声底污染，故默认 T20 区间外推。
    onset_align : bool
        是否先对齐到直达声（peak 截断前面部分）

    返回
    ----
    t60 : float  外推到 -60dB 的 T60（秒），失败返回 nan
    """
    h = np.asarray(h, dtype=np.float64)
    if onset_align:
        onset = int(np.argmax(np.abs(h)))
        h = h[onset:]
    if len(h) < 50:
        return np.nan

    h2 = h ** 2
    cumulative = np.cumsum(h2[::-1])[::-1] + 1e-12
    db = 10.0 * np.log10(cumulative / cumulative.max())
    # 严格用 0 ~ -decay_db 区间回归（不扩大到噪声底）
    idx = np.where(db >= -decay_db)[0]
    if len(idx) < 8:
        return np.nan
    t = idx / sr
    slope, _ = np.polyfit(t, db[idx], 1)
    if slope >= 0:
        return np.nan
    return -60.0 / slope                    # 外推到 T60


# ─── 核心 2：时域指数整形（增广操作）──────────────────────────────

def reshape_t60(h, sr, t60_new, onset_idx=None):
    """时域指数整形：把 RIR 的 T60 改到 t60_new

    原理：h(t) 含 e^{-t/τ_old} 衰减，乘修正因子 e^{-(1/τ_new - 1/τ_old)t}
         衰减率变为 1/τ_new，T60 即被改写。

    参数
    ----
    h : np.ndarray        原始 RIR
    sr : int
    t60_new : float       目标 T60（秒）
    onset_idx : int/None  直达声位置，此点之前不整形；None 自动检测

    返回
    ----
    h_new : np.ndarray    整形后 RIR
    """
    h = np.asarray(h, dtype=np.float64)
    t60_old = schroeder_t60(h, sr, decay_db=20)
    if np.isnan(t60_old):
        raise ValueError("无法估计原 T60（动态范围不足或波形异常）")

    tau_old = t60_old / (6.0 * np.log(10.0))
    tau_new = t60_new / (6.0 * np.log(10.0))

    t = np.arange(len(h)) / sr
    correction = np.exp(-(1.0 / tau_new - 1.0 / tau_old) * t)

    if onset_idx is None:
        onset_idx = _detect_onset(h)
    correction[:onset_idx] = 1.0

    h_new = h * correction
    h_new = h_new * (np.linalg.norm(h) / (np.linalg.norm(h_new) + 1e-12))
    return h_new


def _detect_onset(h, threshold=0.05):
    peak = np.max(np.abs(h))
    above = np.where(np.abs(h) >= threshold * peak)[0]
    return above[0] if len(above) > 0 else 0


def parse_label_t60(filename):
    """从文件名 T60_0.128.wav / T60_0.128_000.wav 提取 T60"""
    m = re.match(r'T60_([0-9]+\.[0-9]+)', filename)
    return float(m.group(1)) if m else None


# ─── 验证流程 ─────────────────────────────────────────────────────

def verify_one(h, sr):
    """返回 {T10,T20,T30} 三种区间外推的 T60"""
    out = {}
    for d in (10, 20, 30):
        out[d] = schroeder_t60(h, sr, decay_db=d)
    return out


def main():
    ap = argparse.ArgumentParser(description="验证 T60 反算精度（真实 RIR 守门脚本）")
    ap.add_argument("--rir_dir", required=True, help="真实 RIR 目录（.wav）")
    ap.add_argument("--out", default="verify_t60_report.csv", help="输出 CSV 报告")
    ap.add_argument("--label_range", default="0.1,1.5",
                    help="只统计标注 T60 在此范围内的样本（避免极端值）")
    args = ap.parse_args()

    lo, hi = [float(x) for x in args.label_range.split(",")]
    wavs = sorted([f for f in os.listdir(args.rir_dir) if f.endswith(".wav")])
    if not wavs:
        print(f"[错误] {args.rir_dir} 下没有 .wav 文件"); return

    rows = []
    by_decay = {10: [], 20: [], 30: []}
    for f in wavs:
        lab = parse_label_t60(f)
        if lab is None:
            continue
        h, sr = sf.read(os.path.join(args.rir_dir, f))
        if h.ndim > 1:
            h = h[:, 0]                      # 多声道取第一声道
        est = verify_one(h, sr)
        row = {"file": f, "label": lab}
        for d in (10, 20, 30):
            e = est[d]
            err = abs(e - lab) / lab * 100 if not np.isnan(e) else float('nan')
            row[f"T{d}_est"] = e
            row[f"T{d}_err%"] = err
            if not np.isnan(err) and lo <= lab <= hi:
                by_decay[d].append(err)
        rows.append(row)

    # 输出明细
    print(f"{'文件':<22}{'标注':>7}{'T20推T60':>10}{'err%':>8}")
    print("-" * 49)
    for r in rows[:12]:
        e = r['T20_est'] if not np.isnan(r['T20_est']) else float('nan')
        err = r['T20_err%']
        print(f"{r['file']:<22}{r['label']:>7.3f}{e:>10.3f}{err:>7.1f}%")
    if len(rows) > 12:
        print(f"... 共 {len(rows)} 条，前 12 条明细如上")

    # 汇总
    print("\n" + "=" * 56)
    print(f"验证汇总（标注 T60 ∈ [{lo}, {hi}]，外推到 T60）")
    for d in (10, 20, 30):
        if by_decay[d]:
            a = np.array(by_decay[d])
            print(f"  T{d} 区间外推: 中位 {np.median(a):5.1f}% | 均 {np.mean(a):5.1f}%"
                  f" | <10%占 {np.mean(a < 10) * 100:3.0f}% | <20%占 {np.mean(a < 20) * 100:3.0f}%")
    print("=" * 56)
    med = np.median(by_decay[20]) if by_decay[20] else 100
    if med < 10:
        print("✅ 结论：标签重算可靠（T20 外推中位误差 <10%），方案成立")
    elif med < 25:
        print("⚠️  结论：可用但需声明误差，或考虑保留原标签策略")
    else:
        print("❌ 结论：Schroeder 重算不可靠，建议保留原 RIR 标签（不重算）")

    # CSV
    import csv
    with open(args.out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "label_t60", "T10_est", "T10_err%",
                    "T20_est", "T20_err%", "T30_est", "T30_err%"])
        for r in rows:
            w.writerow([r["file"], r["label"],
                        r["T10_est"], r["T10_err%"],
                        r["T20_est"], r["T20_err%"],
                        r["T30_est"], r["T30_err%"]])
    print(f"\n[输出] 明细已存 {args.out}")


if __name__ == "__main__":
    main()
