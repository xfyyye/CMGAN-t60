"""
Gradient Remedy (GR) — 抽自 ESPnet GR 仓库,适配本项目的多任务场景。

参考: trainer_gradient_remedy.py (455-475行)
原场景: SE(辅助, enh_model) + ASR(主)
本场景: 去噪(辅助, den) + T60(主) → GR 作用于【共享参数】(encoder + TSCB)

GR 两步:
  1. Projection: 若 g_den 与 g_t60 冲突(dot<0), 把 g_den 投影到不冲突方向,
                 再沿 g_t60 方向补一个分量,使其转为"辅助"。
  2. Rescale:    若 g_den 幅度比 g_t60 大超过 K 倍, 按 cos_theta' 压缩 g_den / 拉伸 g_t60,
                 防止辅助梯度喧宾夺主。

用法(训练循环里):
    optimizer.zero_grad()
    scaler.scale(L_den).backward(retain_graph=True)   # 先 backward 辅助任务
    den_grads = collect_shared_grads(model, shared_prefixes)
    optimizer.zero_grad()
    scaler.scale(L_t60).backward()                     # 再 backward 主任务
    apply_gradient_remedy(model, shared_prefixes, den_grads, K, scaler, optimizer)
    # 现在 .grad 已是 remedy 后的 (g_den + g_t60)
    clip + step
"""

from typing import Dict, List, Iterable

import torch
import torch.nn as nn


# ─── 共享参数识别 ──────────────────────────────────────────────

# 与 TSCNet_KAN_MultiTask_2TSCB 的子模块名对应;
# 共享层 = encoder + 两个 TSCB,其余(denoise decoder / t60 head)为任务专属。
SHARED_PREFIXES_2TSCB = ("dense_encoder", "TSCB_1", "TSCB_2")
# 4-TSCB 版本(备用)
SHARED_PREFIXES_4TSCB = ("dense_encoder", "TSCB_1", "TSCB_2", "TSCB_3", "TSCB_4")


def get_shared_param_names(model: nn.Module, n_tscb: int = 2) -> List[str]:
    """返回共享层参数的完整 name 列表(用于按名取 .grad)。"""
    prefixes = SHARED_PREFIXES_2TSCB if n_tscb == 2 else SHARED_PREFIXES_4TSCB
    return [name for name, _ in model.named_parameters()
            if name.split(".")[0] in prefixes]


# ─── GR 核心 ──────────────────────────────────────────────────

@torch.no_grad()
def _remedy_one_pair(
    g_den_flat: torch.Tensor,
    g_t60_flat: torch.Tensor,
    K: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """对单个参数的 (g_den, g_t60) 做 projection + rescale,返回合并梯度(1-D)。

    与 ESPnet GR 实现(line 457-475)一一对应:
      - dot<0 时 projection: 去掉冲突分量,再补一个 |g_den|/tanθ 的辅助分量
      - |g_den|/|g_t60| > K 且 cos'≠0 时 rescale
    """
    # 跳过退化情况(某一方为零)
    den_norm = g_den_flat.norm()
    t60_norm = g_t60_flat.norm()
    if den_norm < eps or t60_norm < eps:
        return g_den_flat + g_t60_flat

    tan_theta = den_norm / t60_norm
    t60_unit = g_t60_flat / t60_norm

    dot = torch.dot(g_den_flat, g_t60_flat)

    # 1. Gradient Projection
    if dot < 0:
        # 去掉 g_den 在 g_t60 方向的(负)分量
        g_den_flat = g_den_flat - dot * g_t60_flat / (t60_norm ** 2)
        # 沿 g_t60 方向补一个辅助分量,使其转为协助方向
        g_den_flat = g_den_flat + (g_den_flat.norm() / tan_theta) * t60_unit

    # 2. Gradient Rescale
    den_mag = g_den_flat.norm()
    t60_mag = g_t60_flat.norm()
    if den_mag < eps or t60_mag < eps:
        return g_den_flat + g_t60_flat
    cos_prime = torch.dot(g_den_flat, g_t60_flat) / (den_mag * t60_mag)
    if (den_mag / t60_mag) > K and not torch.isclose(cos_prime, torch.tensor(0.0, device=g_den_flat.device)):
        g_den_flat = g_den_flat * cos_prime      # 压缩辅助梯度
        g_t60_flat = g_t60_flat / cos_prime      # 拉伸主梯度

    return g_den_flat + g_t60_flat


@torch.no_grad()
def apply_gradient_remedy(
    model: nn.Module,
    shared_names: List[str],
    den_grads: Dict[str, torch.Tensor],
    K: float = 5.0,
    scaler=None,
    optimizer=None,
):
    """在共享参数上应用 GR,把 .grad 改写为 remedy 后的合并梯度。

    调用时机: 已分别对 L_den 和 L_t60 各做一次 backward(L_t60 的梯度当前在 .grad 里),
              den_grads 是 L_den backward 后采集的梯度快照。

    混合精度: 若提供 scaler/optimizer,先 unscale_ 使 .grad 还原到真实尺度再做投影。
              (scaler 缩放是线性的,cos 符号与比例判断不受影响,但 K 比较需真实尺度。)
    """
    # 混合精度下先 unscale,保证幅度比较正确
    if scaler is not None and optimizer is not None:
        scaler.unscale_(optimizer)

    name_to_param = dict(model.named_parameters())
    for name in shared_names:
        p = name_to_param[name]
        if p.grad is None:
            continue
        g_t60 = p.grad.data           # 当前是 L_t60 backward 留下的
        g_den = den_grads.get(name)
        if g_den is None:
            continue
        merged = _remedy_one_pair(g_den.view(-1).clone(), g_t60.view(-1).clone(), K)
        p.grad.data = merged.view_as(p.grad).to(p.device)


@torch.no_grad()
def collect_shared_grads(model: nn.Module, shared_names: List[str]) -> Dict[str, torch.Tensor]:
    """采集当前 .grad 的快照(用于 L_den backward 后保存去噪梯度)。"""
    name_to_param = dict(model.named_parameters())
    out = {}
    for name in shared_names:
        p = name_to_param[name]
        if p.grad is not None:
            out[name] = p.grad.data.clone()
    return out
