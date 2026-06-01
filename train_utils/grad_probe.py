"""
GradAngleProbe: 观察多任务训练中共享 backbone 上各任务梯度的夹角与量级。

设计:
- 按模块分组（encoder / TSCB_1 / TSCB_2 [/ TSCB_3 / TSCB_4]）分别计算
- 必须在 backward 之前调用；内部用 autograd.grad + retain_graph，不消耗主路径计算图
- 兼容 AMP GradScaler: 自动用 scaler.scale 包装，再除以 get_scale() 还原
- 每 log_every 个 train step 触发一次以控制开销

返回指标 (每组):
- cos                : 两任务梯度余弦相似度，<0 即冲突
- norm_denoise       : 去噪梯度 L2 范数
- norm_t60           : T60 梯度 L2 范数
- ratio_d_over_t     : ‖g_denoise‖ / ‖g_t60‖，量级失衡度
"""

import torch


class GradAngleProbe:
    EPS = 1e-12

    def __init__(self, raw_model, log_every=20, include_shared_all=False):
        """
        Args:
            raw_model: TSCNet_MultiTask 或 TSCNet_MultiTask_2TSCB（**未** 经 DataParallel 包装）
            log_every: 每多少个 train step 触发一次（按 step_idx 计数）
            include_shared_all: 额外计算所有共享参数串起来的整体 cos / norm（开销近乎翻倍）
        """
        self.groups = {
            'encoder': list(raw_model.dense_encoder.parameters()),
            'tscb_1':  list(raw_model.TSCB_1.parameters()),
        }
        for name in ('TSCB_2', 'TSCB_3', 'TSCB_4'):
            if hasattr(raw_model, name):
                self.groups[name.lower()] = list(getattr(raw_model, name).parameters())

        if include_shared_all:
            self.groups['shared_all'] = sum(self.groups.values(), [])

        self.log_every = log_every

    def should_log(self, step_idx):
        return (step_idx % self.log_every) == 0

    @staticmethod
    def _flatten(grads, params):
        vs = []
        for g, p in zip(grads, params):
            if g is None:
                vs.append(torch.zeros_like(p, dtype=torch.float32).flatten())
            else:
                vs.append(g.detach().float().flatten())
        return torch.cat(vs)

    def measure(self, loss_denoise, loss_t60, scaler=None):
        """对当前计算图分别求 denoise / t60 的梯度。必须在主 backward 之前调用。

        Args:
            loss_denoise: scalar tensor，去噪 loss（带计算图）
            loss_t60:     scalar tensor，T60 loss（带计算图）
            scaler:       torch.cuda.amp.GradScaler 或 None

        Returns:
            dict[str, float]
        """
        scale = scaler.get_scale() if scaler is not None else 1.0
        loss_d = scaler.scale(loss_denoise) if scaler is not None else loss_denoise
        loss_t = scaler.scale(loss_t60)     if scaler is not None else loss_t60

        out = {}
        for name, params in self.groups.items():
            if len(params) == 0:
                continue
            g_d = torch.autograd.grad(
                loss_d, params,
                retain_graph=True, allow_unused=True,
            )
            g_t = torch.autograd.grad(
                loss_t, params,
                retain_graph=True, allow_unused=True,
            )
            vd = self._flatten(g_d, params) / scale
            vt = self._flatten(g_t, params) / scale
            nd = vd.norm()
            nt = vt.norm()
            cos = (vd @ vt) / (nd * nt + self.EPS)

            out[f'grad/{name}/cos']             = cos.item()
            out[f'grad/{name}/norm_denoise']    = nd.item()
            out[f'grad/{name}/norm_t60']        = nt.item()
            out[f'grad/{name}/ratio_d_over_t']  = (nd / (nt + self.EPS)).item()

            del g_d, g_t, vd, vt
        return out
