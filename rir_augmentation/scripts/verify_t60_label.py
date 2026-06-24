"""
T60 物理增广 —— 标签精度验证脚本（文档 3）

【作用】
验证"时域指数整形改变 T60"这一物理变换是否成立。
具体：对一条真实 RIR 估原 T60 → 整形到目标 T60' → Schroeder 反算新 T60'，
      看反算值与理论值的误差是否 < 5%。

【守门规则】
若多条 RIR 上平均误差 < 5%  → 方案成立，可进入完整实现
若误差 5%-10%              → 需排查（直达声截断/噪声尾/频段限制）
若误差 > 10%               → 方案返工，需重新设计整形

【用法】（待原始数据就绪后）
    python verify_t60_label.py --rir_dir data/rir_real --out report.csv

【依赖】
    pip install numpy soundfile
"""

import os
import argparse
import numpy as np
import soundfile as sf


# ─── 核心 1：Schroeder 后向积分估 T60 ──────────────────────────────

def schroeder_t60(h, sr, decay_db=60, reg_factor=1e-10):
    """从 RIR 波形估 T60（Schroeder 后向积分 + 线性回归）

    参数
    ----
    h : np.ndarray, (N,)
        RIR 波形
    sr : int
        采样率
    decay_db : int
        衰减区间（默认 60 → T60）
    reg_factor : float
        防止 log(0) 的小正数

    返回
    ----
    t60 : float
        估计的混响时间（秒）
    """
    h = np.asarray(h, dtype=np.float64)
    h2 = h ** 2                                  # 能量
    # 后向积分：L(t) = ∫_t^T h²(τ)dτ
    cumulative = np.cumsum(h2[::-1])[::-1]
    # 跳过极小值尾部，避免噪声污染回归
    cumulative += reg_factor
    db = 10.0 * np.log10(cumulative / cumulative.max())
    # 在 0 ~ -decay_db 区间取点
    idx = np.where(db >= -decay_db)[0]
    if len(idx) < 10:                            # 点太少，无法回归
        return np.nan
    t = idx / sr
    slope, _ = np.polyfit(t, db[idx], 1)
    if slope >= 0:                               # 衰减斜率必须为负
        return np.nan
    t60 = -decay_db / slope                      # 外推到 -60dB 的时间
    return t60


# ─── 核心 2：时域指数整形（增广操作）──────────────────────────────

def reshape_t60(h, sr, t60_new, onset_idx=None):
    """时域指数整形：把 RIR 的 T60 改到 t60_new

    原理：h(t) 含 e^{-t/τ_old} 衰减，乘修正因子 e^{-(1/τ_new - 1/τ_old)t}
         衰减率变为 1/τ_new，T60 即被改写。

    参数
    ----
    h : np.ndarray
        原始 RIR
    sr : int
    t60_new : float
        目标 T60（秒）
    onset_idx : int or None
        直达声位置，此点之前不整形（保护早期反射）
        None 表示自动检测

    返回
    ----
    h_new : np.ndarray  整形后 RIR
    """
    h = np.asarray(h, dtype=np.float64)
    # 估原 T60
    t60_old = schroeder_t60(h, sr)
    if np.isnan(t60_old):
        raise ValueError("无法估计原 T60，RIR 质量可能有问题")

    tau_old = t60_old / (6.0 * np.log(10.0))     # τ = T60 / 13.82
    tau_new = t60_new / (6.0 * np.log(10.0))

    # 修正包络
    t = np.arange(len(h)) / sr
    correction = np.exp(-(1.0 / tau_new - 1.0 / tau_old) * t)

    # 保护直达声与早期反射（onset 之前不动）
    if onset_idx is None:
        onset_idx = _detect_onset(h)
    correction[:onset_idx] = 1.0

    h_new = h * correction
    # 能量归一化（对齐训练分布）
    h_new = h_new * (np.linalg.norm(h) / (np.linalg.norm(h_new) + 1e-12))
    return h_new


def _detect_onset(h, threshold=0.05):
    """简单 onset 检测：第一个超过 peak × threshold 的点"""
    peak = np.max(np.abs(h))
    above = np.where(np.abs(h) >= threshold * peak)[0]
    return above[0] if len(above) > 0 else 0


# ─── 验证流程 ─────────────────────────────────────────────────────

def verify_single(h, sr, t60_targets):
    """对一条 RIR 验证多个目标 T60

    返回 [(t60_target, t60_back_estimated, error_pct), ...]
    """
    results = []
    for t60_tgt in t60_targets:
        try:
            h_new = reshape_t60(h, sr, t60_tgt)
            t60_est = schroeder_t60(h_new, sr)
            if np.isnan(t60_est):
                results.append((t60_tgt, np.nan, np.nan))
            else:
                err = abs(t60_est - t60_tgt) / t60_tgt * 100.0
                results.append((t60_tgt, t60_est, err))
        except Exception as e:
            results.append((t60_tgt, np.nan, np.nan))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="验证 T60 物理整形标签精度（文档 3 守门脚本）"
    )
    parser.add_argument("--rir_dir", required=True,
                        help="真实 RIR 目录（.wav）")
    parser.add_argument("--out", default="verify_t60_report.csv",
                        help="输出 CSV 报告")
    parser.add_argument("--n_rir", type=int, default=20,
                        help="抽样验证的 RIR 数量")
    parser.add_argument("--offset", type=float, default=0.15,
                        help="T60 偏移比例（默认 ±15%）")
    args = parser.parse_args()

    # 收集 RIR
    wavs = sorted([f for f in os.listdir(args.rir_dir) if f.endswith(".wav")])
    if not wavs:
        print(f"[错误] {args.rir_dir} 下没有 .wav 文件")
        return
    n = min(args.n_rir, len(wavs))
    sample = np.linspace(0, len(wavs) - 1, n).astype(int)
    print(f"[信息] 抽样 {n} 条 RIR 验证")

    # 目标 T60 偏移（保守）
    all_rows = []
    for i in sample:
        path = os.path.join(args.rir_dir, wavs[i])
        h, sr = sf.read(path)
        if h.ndim > 1:
            h = h.mean(axis=1)
        # 估原 T60
        t60_orig = schroeder_t60(h, sr)
        if np.isnan(t60_orig):
            print(f"  [跳过] {wavs[i]} 无法估原 T60")
            continue
        # 目标 T60：原 × {1-offset, 1+offset}
        targets = sorted({max(0.1, t60_orig * (1 - args.offset)),
                          t60_orig,
                          min(1.5, t60_orig * (1 + args.offset))})
        res = verify_single(h, sr, targets)
        for t_tgt, t_est, err in res:
            all_rows.append([wavs[i], t60_orig, t_tgt, t_est, err])
            tag = "✓" if (not np.isnan(err) and err < 5.0) else "✗"
            print(f"  {wavs[i]}  原T60={t60_orig:.3f}  目标={t_tgt:.3f}  "
                  f"反算={t_est:.3f}  误差={err:.2f}%  {tag}")

    # 汇总
    errs = [r[4] for r in all_rows if not np.isnan(r[4])]
    if errs:
        mean_err = np.mean(errs)
        max_err = np.max(errs)
        pass_rate = np.mean(np.array(errs) < 5.0) * 100.0
        print("\n" + "=" * 50)
        print(f"验证汇总（{len(errs)} 个点）")
        print(f"  平均误差: {mean_err:.2f}%")
        print(f"  最大误差: {max_err:.2f}%")
        print(f"  误差<5% 通过率: {pass_rate:.1f}%")
        print("=" * 50)
        if mean_err < 5.0:
            print("✅ 结论：方案成立，可进入完整实现")
        elif mean_err < 10.0:
            print("⚠️  结论：误差偏高，需排查（直达声截断/噪声尾）")
        else:
            print("❌ 结论：误差过大，方案需返工")

    # 存 CSV
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rir", "t60_original", "t60_target",
                    "t60_back_estimated", "error_pct"])
        w.writerows(all_rows)
    print(f"\n[输出] 报告已存 {args.out}")


if __name__ == "__main__":
    main()
