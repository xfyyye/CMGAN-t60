"""
Gradient Remedy (GR) — 抽自 ESPnet GR 仓库,适配本项目的多任务场景。

参考: trainer_gradient_remedy.py (455-475行)

本场景(默认): 去噪(主) + T60(辅助)
  → GR 修改【辅梯度 g_t60】去配合【主梯度 g_den】
  → 训练循环: 先 backward 去噪(采集 main_grads), 再 backward T60(.grad 是 g_aux)

GR 两步:
  1. Projection: 若 g_aux 与 g_main 冲突(dot<0), 把 g_aux 投影到不冲突方向,
                 再沿 g_main 方向补一个分量,使其转为"辅助"。
  2. Rescale:    若 |g_aux| 幅度比 |g_main| 大超过 K 倍, 按 cos_theta' 压缩 g_aux / 拉伸 g_main,
                 防止辅助梯度喧宾夺主。
"""

from typing import Dict, List

import torch
import torch.nn as nn


# ─── 共享参数识别 ──────────────────────────────────────────────

SHARED_PREFIXES_2TSCB = ("dense_encoder", "TSCB_1", "TSCB_2")
SHARED_PREFIXES_4TSCB = ("dense_encoder", "TSCB_1", "TSCB_2", "TSCB_3", "TSCB_4")


def get_shared_param_names(model: nn.Module, n_tscb: int = 2) -> List[str]:
    """返回共享层参数的完整 name 列表(用于按名取 .grad)。"""
    prefixes = SHARED_PREFIXES_2TSCB if n_tscb == 2 else SHARED_PREFIXES_4TSCB
    return [name for name, _ in model.named_parameters()
            if name.split(".")[0] in prefixes]


# ─── GR 核心 ──────────────────────────────────────────────────

@torch.no_grad()
def _remedy_one_pair(
    g_main_flat: torch.Tensor,
    g_aux_flat: torch.Tensor,
    K: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """对单个参数的 (g_main, g_aux) 做 projection + rescale,返回合并梯度(1-D)。

    与 ESPnet GR 实现(line 457-475)一一对应:
      - dot<0 时 projection: 去掉 g_aux 在 g_main 方向的(负)分量,再补一个辅助分量
      - |g_aux|/|g_main| > K 且 cos'≠0 时 rescale
    """
    # 跳过退化情况(某一方为零)
    main_norm = g_main_flat.norm()
    aux_norm = g_aux_flat.norm()
    if main_norm < eps or aux_norm < eps:
        return g_main_flat + g_aux_flat

    tan_theta = aux_norm / main_norm
    main_unit = g_main_flat / main_norm

    dot = torch.dot(g_aux_flat, g_main_flat)

    # 1. Gradient Projection
    if dot < 0:
        # 去掉 g_aux 在 g_main 方向的(负)分量
        g_aux_flat = g_aux_flat - dot * g_main_flat / (main_norm ** 2)
        # 沿 g_main 方向补一个辅助分量,使其转为协助方向
        g_aux_flat = g_aux_flat + (g_aux_flat.norm() / tan_theta) * main_unit

    # 2. Gradient Rescale
    aux_mag = g_aux_flat.norm()
    main_mag = g_main_flat.norm()
    if aux_mag < eps or main_mag < eps:
        return g_main_flat + g_aux_flat
    cos_prime = torch.dot(g_aux_flat, g_main_flat) / (aux_mag * main_mag)
    if (aux_mag / main_mag) > K and not torch.isclose(cos_prime, torch.tensor(0.0, device=g_aux_flat.device)):
        g_aux_flat = g_aux_flat * cos_prime      # 压缩辅助梯度
        g_main_flat = g_main_flat / cos_prime    # 拉伸主梯度

    return g_main_flat + g_aux_flat


@torch.no_grad()
def apply_gradient_remedy(
    model: nn.Module,
    shared_names: List[str],
    main_grads: Dict[str, torch.Tensor],
    K: float = 5.0,
    scaler=None,
    optimizer=None,
    w_main: float = 1.0,
    w_aux: float = 1.0,
):
    """在共享参数上应用 GR,把 .grad 改写为 remedy 后的合并梯度。

    约定:
      - 当前 .grad 是【辅任务】backward 留下的梯度 (g_aux)
      - main_grads 是【主任务】backward 后采集的快照 (g_main)
    本场景默认: 主=去噪, 辅=T60
      → 调用前先 backward 去噪(采集 main_grads), 再 backward T60(.grad)

    参数:
      w_main, w_aux: 主/辅 loss 在总损失中的权重 (合并时加权,与单次 backward 等价)

    混合精度: 若提供 scaler/optimizer,先 unscale_ 使 .grad 还原到真实尺度再做投影。
    """
    # 混合精度下先 unscale,保证幅度比较正确
    if scaler is not None and optimizer is not None:
        scaler.unscale_(optimizer)

    name_to_param = dict(model.named_parameters())
    for name in shared_names:
        p = name_to_param[name]
        if p.grad is None:
            continue
        g_aux = p.grad.data * w_aux         # 当前 .grad 是辅任务梯度
        g_main = main_grads.get(name)
        if g_main is None:
            continue
        g_main = g_main * w_main
        merged = _remedy_one_pair(g_main.view(-1).clone(), g_aux.view(-1).clone(), K)
        p.grad.data = merged.view_as(p.grad).to(p.device)


@torch.no_grad()
def collect_shared_grads(model: nn.Module, shared_names: List[str]) -> Dict[str, torch.Tensor]:
    """采集当前 .grad 的快照(用于主任务 backward 后保存主梯度)。"""
    name_to_param = dict(model.named_parameters())
    out = {}
    for name in shared_names:
        p = name_to_param[name]
        if p.grad is not None:
            out[name] = p.grad.data.clone()
    return out
