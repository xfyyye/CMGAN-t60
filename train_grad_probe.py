"""
CMGAN 多任务训练 + 梯度夹角探针: T60估计 + 去噪

基于 master/train.py 改造:
- MultiTaskLoss.forward 额外返回带计算图的 denoise_loss / t60_loss tensor
- 每 args.probe_log_every 个 train step, 在主 backward 之前调用 GradAngleProbe
- 探针指标 (cos / norm_denoise / norm_t60 / ratio) 推送到 SwanLab, step=global_step

其余逻辑（数据加载、AMP、梯度累积、早停、LR 调度）与 master/train.py 一致。
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generator_t60 import TSCNet_MultiTask, TSCNet_MultiTask_2TSCB
from dataset import DEFAULT_DATASET_ROOT, create_dataloaders, resolve_dataset_root, T60Normalizer
from utils import power_compress, power_uncompress
from train_utils.grad_probe import GradAngleProbe


# ─── 多任务损失（暴露分项 tensor） ────────────────────────────────

class MultiTaskLossWithGrads(nn.Module):
    """与 train.MultiTaskLoss 数学等价，但 forward 额外返回带 grad 的 denoise/t60 tensor。"""

    def __init__(
        self,
        n_fft=400, hop=100,
        w_ri=0.1, w_mag=0.9, w_time=0.2,
        alpha=0.1, beta=1.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.w_ri = w_ri
        self.w_mag = w_mag
        self.w_time = w_time
        self.alpha = alpha
        self.beta = beta

    def forward(self, est_real, est_imag, clean_spec, t60_pred, t60_target, clean_wav=None):
        clean_pc = power_compress(clean_spec.permute(0, 3, 2, 1))
        clean_real = clean_pc[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_pc[:, 1, :, :].unsqueeze(1)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

        est_real_f = est_real.permute(0, 1, 3, 2)
        est_imag_f = est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real_f ** 2 + est_imag_f ** 2)

        loss_ri = F.mse_loss(est_real_f, clean_real) + F.mse_loss(est_imag_f, clean_imag)
        loss_mag = F.mse_loss(est_mag, clean_mag)

        loss_time = torch.tensor(0.0, device=est_real.device)
        if self.w_time > 0 and clean_wav is not None:
            est_spec_uncompress = power_uncompress(est_real_f, est_imag_f).squeeze(1)
            est_spec_complex = torch.complex(est_spec_uncompress[..., 0], est_spec_uncompress[..., 1])
            est_audio = torch.istft(
                est_spec_complex,
                self.n_fft, self.hop,
                window=torch.hamming_window(self.n_fft).to(est_real.device),
                onesided=True,
            )
            min_len = min(est_audio.size(-1), clean_wav.size(-1))
            loss_time = F.l1_loss(est_audio[..., :min_len], clean_wav[..., :min_len])

        denoise_loss = self.w_ri * loss_ri + self.w_mag * loss_mag + self.w_time * loss_time
        t60_loss = F.mse_loss(t60_pred, t60_target)
        total = self.alpha * denoise_loss + self.beta * t60_loss

        metrics = {
            'denoise_loss': denoise_loss.item(),
            'loss_ri': loss_ri.item(),
            'loss_mag': loss_mag.item(),
            'loss_time': loss_time.item(),
            't60_loss': t60_loss.item(),
            'total_loss': total.item(),
        }
        return total, denoise_loss, t60_loss, metrics


# ─── 早停 ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, mode='min', min_delta=1e-4, save_path='best_model.pth'):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save(model)
            return False
        improved = (score < self.best_score - self.min_delta) if self.mode == 'min' \
            else (score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
            self._save(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def _save(self, model):
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({'model_state_dict': raw_model.state_dict(), 'best_score': self.best_score}, self.save_path)


# ─── 主训练逻辑 ──────────────────────────────────────────────────

def train(args):
    args.dataset_root = resolve_dataset_root(args.dataset_root)
    print(f'数据集路径: {args.dataset_root}')
    print(f'固定音频长度: {args.audio_length:.2f}s ({int(args.audio_length * args.target_sr)} samples)')
    print(f'梯度探针: 每 {args.probe_log_every} 个 train step 触发一次')

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0')
    print(f'可用 GPU: {n_gpu} 张 ({[torch.cuda.get_device_name(i) for i in range(n_gpu)]})')

    try:
        import swanlab
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.experiment_name,
            config=vars(args),
        )
        use_swanlab = True
    except Exception as e:
        print(f'SwanLab 初始化失败 ({e}), 不使用实验追踪')
        use_swanlab = False
        swanlab_run = None

    train_loader, eval_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_length=args.audio_length,
        target_sr=args.target_sr,
    )
    print(f'训练集: {len(train_loader.dataset)} 样本, 验证集: {len(eval_loader.dataset)} 样本')

    if args.n_tscb == 2:
        model = TSCNet_MultiTask_2TSCB(num_channel=64, num_features=args.n_fft // 2 + 1)
    else:
        model = TSCNet_MultiTask(num_channel=64, num_features=args.n_fft // 2 + 1)
    model = model.to(device)
    raw_model = model
    if n_gpu > 1:
        model = nn.DataParallel(model)
        print(f'使用 DataParallel × {n_gpu} GPU')
    raw_params = sum(p.numel() for p in raw_model.parameters())
    print(f'模型参数: {raw_params:,} ({raw_params / 1e6:.2f}M)')

    criterion = MultiTaskLossWithGrads(
        n_fft=args.n_fft, hop=args.hop_length,
        w_ri=args.w_ri, w_mag=args.w_mag, w_time=args.w_time,
        alpha=args.alpha, beta=args.beta,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    os.makedirs(args.save_dir, exist_ok=True)
    early_stop = EarlyStopping(
        patience=args.early_stop_patience, mode='min',
        save_path=os.path.join(args.save_dir, 'best_model.pth'),
    )

    t60_normalizer = T60Normalizer(args.t60_min, args.t60_max)

    probe = GradAngleProbe(
        raw_model=raw_model,
        log_every=args.probe_log_every,
        include_shared_all=args.probe_include_shared_all,
    )
    print(f'探针监控模块: {list(probe.groups.keys())}')

    history = []
    print(f'\n{"Epoch":>6} {"Train Loss":>10} {"Val Loss":>10} {"Denoise":>10} {"T60":>10} {"LR":>12}')
    print('─' * 66)

    accum_steps = args.accum_steps
    effective_batch = args.batch_size * accum_steps
    print(f'实际 batch={args.batch_size}, 累积 {accum_steps} 步, 等效 batch={effective_batch}')

    train_total = len(train_loader.dataset)
    val_total = len(eval_loader.dataset)
    global_step = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_sum = 0
        train_metrics_sum = {}
        train_count = 0
        train_samples = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            noisy_spec = batch['noisy_spec'].to(device)
            clean_spec = batch['clean_spec'].to(device)
            clean_wav = batch['clean_wav'].to(device)
            t60_target = batch['t60'].to(device)

            noisy_input = power_compress(noisy_spec.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)

            with torch.amp.autocast('cuda'):
                est_real, est_imag, t60_pred = model(noisy_input)
                loss, denoise_loss, t60_loss, metrics = criterion(
                    est_real, est_imag, clean_spec,
                    t60_pred, t60_target,
                    clean_wav=clean_wav,
                )
                scaled_loss = loss / accum_steps

            # ── 梯度探针 (在主 backward 之前) ──
            if probe.should_log(global_step):
                probe_metrics = probe.measure(denoise_loss, t60_loss, scaler=scaler)
                if use_swanlab:
                    swanlab_run.log(probe_metrics, step=global_step)
                # 同时打印到 stdout，方便 tail -f nohup.log 直接看冲突情况
                probe_parts = [
                    f'{g}:cos={probe_metrics[f"grad/{g}/cos"]:+.3f},r={probe_metrics[f"grad/{g}/ratio_d_over_t"]:.2f}'
                    for g in probe.groups.keys()
                ]
                sys.stdout.write(f'\n[probe step={global_step}] ' + ' '.join(probe_parts) + '\n')
                sys.stdout.flush()

            scaler.scale(scaled_loss).backward()

            train_loss_sum += metrics['total_loss']
            for k, v in metrics.items():
                train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
            train_count += 1
            train_samples += noisy_spec.size(0)
            global_step += 1

            if (step_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pct = train_samples / train_total * 100
            sys.stdout.write(
                f'\r  Epoch {epoch} 训练: {train_samples}/{train_total} ({pct:.1f}%) '
                f'loss={metrics["total_loss"]:.4f} t60={metrics["t60_loss"]:.4f}'
            )
            sys.stdout.flush()

        avg_train_loss = train_loss_sum / train_count
        avg_train_metrics = {k: v / train_count for k, v in train_metrics_sum.items()}

        # ── 验证 ──
        model.eval()
        val_loss_sum = 0
        val_metrics_sum = {}
        val_count = 0
        val_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                noisy_spec = batch['noisy_spec'].to(device)
                clean_spec = batch['clean_spec'].to(device)
                clean_wav = batch['clean_wav'].to(device)
                t60_target = batch['t60'].to(device)

                noisy_input = power_compress(noisy_spec.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)

                with torch.cuda.amp.autocast():
                    est_real, est_imag, t60_pred = model(noisy_input)
                    loss, _, _, metrics = criterion(
                        est_real, est_imag, clean_spec,
                        t60_pred, t60_target,
                        clean_wav=clean_wav,
                    )

                val_loss_sum += metrics['total_loss']
                for k, v in metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0) + v
                val_count += 1
                val_samples += noisy_spec.size(0)

                pct = val_samples / val_total * 100
                sys.stdout.write(
                    f'\r  Epoch {epoch} 验证: {val_samples}/{val_total} ({pct:.1f}%) '
                    f'loss={metrics["total_loss"]:.4f}'
                )
                sys.stdout.flush()

        avg_val_loss = val_loss_sum / val_count
        avg_val_metrics = {k: v / val_count for k, v in val_metrics_sum.items()}

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        early_stop(avg_val_loss, model)
        is_best = early_stop.counter == 0 and early_stop.best_score == avg_val_loss
        marker = ' *' if is_best else ''

        print(f'{epoch:>6d} {avg_train_loss:>10.4f} {avg_val_loss:>10.4f} '
              f'{avg_val_metrics.get("denoise_loss", 0):>10.4f} '
              f'{avg_val_metrics.get("t60_loss", 0):>10.4f} '
              f'{current_lr:>12.2e}{marker}')

        if use_swanlab:
            log_dict = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr,
                'epoch': epoch,
            }
            for k, v in avg_train_metrics.items():
                log_dict[f'train_{k}'] = v
            for k, v in avg_val_metrics.items():
                log_dict[f'val_{k}'] = v
            swanlab_run.log(log_dict, step=global_step)

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            **{f'train_{k}': v for k, v in avg_train_metrics.items()},
            **{f'val_{k}': v for k, v in avg_val_metrics.items()},
        })

        if early_stop.should_stop:
            print(f'\n早停! 最佳验证损失: {early_stop.best_score:.6f}')
            break

    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(early_stop.save_path)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    print(f'已加载最佳模型: {early_stop.save_path}')

    if use_swanlab:
        swanlab_run.finish()

    return model


# ─── 入口 ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='CMGAN 多任务训练 + 梯度夹角探针')

    # 数据
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--n_fft', type=int, default=400)
    parser.add_argument('--hop_length', type=int, default=100)
    parser.add_argument('--audio_length', type=float, default=4.0)
    parser.add_argument('--target_sr', type=int, default=16000)
    parser.add_argument('--t60_min', type=float, default=0.1)
    parser.add_argument('--t60_max', type=float, default=1.5)

    # 训练
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accum_steps', type=int, default=2, help='梯度累积步数')
    parser.add_argument('--n_tscb', type=int, default=4, choices=[2, 4], help='TSCB层数')

    # 损失权重
    parser.add_argument('--w_ri', type=float, default=0.1)
    parser.add_argument('--w_mag', type=float, default=0.9)
    parser.add_argument('--w_time', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1, help='去噪损失权重')
    parser.add_argument('--beta', type=float, default=1.0, help='T60损失权重')

    # 梯度探针
    parser.add_argument('--probe_log_every', type=int, default=20,
                        help='每多少个 train step 触发一次梯度夹角观测')
    parser.add_argument('--probe_include_shared_all', action='store_true',
                        help='额外计算所有共享参数串起来的整体 cos/norm (开销近乎翻倍)')

    # 输出
    parser.add_argument('--save_dir', type=str, default='runs')
    parser.add_argument('--experiment_name', type=str, default='CMGAN_t60_grad_probe')
    parser.add_argument('--swanlab_project', type=str, default='T60_GradAngleProbe')

    # GPU
    parser.add_argument('--gpu_id', type=int, default=0)

    # 模式
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.test_only:
        model = train(args)

        if args.train_only:
            print('训练完成 (--train_only 模式，跳过测试)')
            exit(0)

        args.model_path = os.path.join(args.save_dir, 'best_model.pth')

    if args.model_path:
        from test import run_tests
        run_tests(args)
