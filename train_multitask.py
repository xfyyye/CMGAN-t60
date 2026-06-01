"""
CMGAN KAN-head 多任务训练: T60估计 + 去噪

基于 train.py (yaml 驱动) 扩展:
- 模型: TSCNet_KAN_MultiTask{_2TSCB}, 共享 backbone + KAN head + 去噪 decoder
- 损失: alpha * denoise + beta * t60_loss
       T60 支持 mse / mae / huber
       Denoise 沿用 CMGAN 标准的 RI + Magnitude + 时域 L1
- 梯度夹角探针: 每 probe_log_every 个 train step 在主 backward 之前测量
                cos / norm_denoise / norm_t60 / ratio (per module group)
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generator_t60 import (
    TSCNet_KAN_MultiTask,
    TSCNet_KAN_MultiTask_2TSCB,
)
from dataset import (
    DEFAULT_DATASET_ROOT,
    create_multitask_dataloaders,
    resolve_dataset_root,
)
from utils import power_compress, power_uncompress
from train_utils.grad_probe import GradAngleProbe


# ─── 多任务损失 (暴露分项 tensor 给梯度探针) ────────────────────────

class MultiTaskLossWithGrads(nn.Module):
    """alpha * denoise + beta * t60 ，forward 额外返回带 grad 的分项 tensor。"""

    def __init__(
        self,
        n_fft=400, hop=100,
        w_ri=0.1, w_mag=0.9, w_time=0.2,
        alpha=0.1, beta=1.0,
        t60_loss_type='mse', huber_delta=1.0,
    ):
        super().__init__()
        if t60_loss_type not in {'mse', 'mae', 'huber'}:
            raise ValueError(f'Unsupported t60_loss_type: {t60_loss_type}')
        self.n_fft = n_fft
        self.hop = hop
        self.w_ri = w_ri
        self.w_mag = w_mag
        self.w_time = w_time
        self.alpha = alpha
        self.beta = beta
        self.t60_loss_type = t60_loss_type
        self.huber_delta = huber_delta

    def _t60_loss(self, t60_pred, t60_target):
        if self.t60_loss_type == 'mse':
            return F.mse_loss(t60_pred, t60_target)
        if self.t60_loss_type == 'mae':
            return F.l1_loss(t60_pred, t60_target)
        return F.huber_loss(t60_pred, t60_target, delta=self.huber_delta)

    def forward(self, est_real, est_imag, clean_spec, t60_pred, t60_target, clean_wav=None):
        # clean_spec: (B, 2, T, F) 原始 STFT；先做与输入一致的功率压缩
        clean_pc = power_compress(clean_spec.permute(0, 3, 2, 1))
        clean_real = clean_pc[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_pc[:, 1, :, :].unsqueeze(1)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

        # 模型输出 est_real/est_imag: (B, 1, T, F) → (B, 1, F, T) 与 clean 对齐
        est_real_f = est_real.permute(0, 1, 3, 2)
        est_imag_f = est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real_f ** 2 + est_imag_f ** 2)

        loss_ri = F.mse_loss(est_real_f, clean_real) + F.mse_loss(est_imag_f, clean_imag)
        loss_mag = F.mse_loss(est_mag, clean_mag)

        loss_time = torch.tensor(0.0, device=est_real.device)
        if self.w_time > 0 and clean_wav is not None:
            est_spec_uncompress = power_uncompress(est_real_f, est_imag_f).squeeze(1)
            est_spec_complex = torch.complex(
                est_spec_uncompress[..., 0], est_spec_uncompress[..., 1]
            )
            est_audio = torch.istft(
                est_spec_complex,
                self.n_fft, self.hop,
                window=torch.hamming_window(self.n_fft).to(est_real.device),
                onesided=True,
            )
            min_len = min(est_audio.size(-1), clean_wav.size(-1))
            loss_time = F.l1_loss(est_audio[..., :min_len], clean_wav[..., :min_len])

        denoise_loss = self.w_ri * loss_ri + self.w_mag * loss_mag + self.w_time * loss_time
        t60_loss = self._t60_loss(t60_pred, t60_target)
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
        torch.save(
            {'model_state_dict': raw_model.state_dict(), 'best_score': self.best_score},
            self.save_path,
        )


# ─── 主训练逻辑 ──────────────────────────────────────────────────

def train(args):
    args.dataset_root = resolve_dataset_root(args.dataset_root)
    if args.config:
        print(f'配置文件: {args.config}')
    print(f'数据集路径: {args.dataset_root}')
    print(f'固定音频长度: {args.audio_length:.2f}s ({int(args.audio_length * args.target_sr)} samples)')
    print(f'多任务权重: alpha(denoise)={args.alpha} beta(t60)={args.beta}')
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

    train_loader, eval_loader = create_multitask_dataloaders(
        dataset_root=args.dataset_root,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_length=args.audio_length,
        target_sr=args.target_sr,
        log_scale=args.log_scale,
    )
    print(f'训练集: {len(train_loader.dataset)} 样本, 验证集: {len(eval_loader.dataset)} 样本')

    fourier_hidden_dims = getattr(args, 't60_fourier_hidden_dims', (32, 16))

    model_kwargs = dict(
        num_channel=64,
        num_features=args.n_fft // 2 + 1,
        t60_head_type=args.t60_head_type,
        t60_hidden_dim=args.t60_hidden_dim,
        t60_dropout=args.t60_dropout,
        t60_fourier_proj_dim=args.t60_fourier_proj_dim,
        t60_fourier_hidden_dims=fourier_hidden_dims,
        t60_fourier_first_num_frequencies=args.t60_fourier_first_num_frequencies,
        t60_fourier_hidden_num_frequencies=args.t60_fourier_hidden_num_frequencies,
        t60_fourier_dropout=args.t60_fourier_dropout,
        t60_out_activation=args.t60_out_activation,
    )
    if args.n_tscb == 2:
        model = TSCNet_KAN_MultiTask_2TSCB(**model_kwargs)
    else:
        model = TSCNet_KAN_MultiTask(**model_kwargs)
    model = model.to(device)
    raw_model = model
    if n_gpu > 1:
        model = nn.DataParallel(model)
        print(f'使用 DataParallel × {n_gpu} GPU')
    raw_params = sum(p.numel() for p in raw_model.parameters())
    head_params = sum(p.numel() for p in raw_model.t60_head.parameters())
    print(f'T60 head 类型: {args.t60_head_type}')
    print(f'T60 head 参数: {head_params:,}')
    print(f'模型总参数: {raw_params:,} ({raw_params / 1e6:.2f}M)')

    criterion = MultiTaskLossWithGrads(
        n_fft=args.n_fft, hop=args.hop_length,
        w_ri=args.w_ri, w_mag=args.w_mag, w_time=args.w_time,
        alpha=args.alpha, beta=args.beta,
        t60_loss_type=args.loss, huber_delta=args.huber_delta,
    )
    print(f'T60 loss: {args.loss} | denoise (w_ri, w_mag, w_time)=({args.w_ri}, {args.w_mag}, {args.w_time})')

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

    probe = GradAngleProbe(
        raw_model=raw_model,
        log_every=args.probe_log_every,
        include_shared_all=args.probe_include_shared_all,
    )
    print(f'探针监控模块: {list(probe.groups.keys())}')

    history = []
    print(f'\n{"Epoch":>6} {"Train":>10} {"Val":>10} {"Denoise":>10} {"T60":>10} {"LR":>12}')
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
            noisy_spec = batch['noisy_spec'].to(device)   # (B, 2, T, F)
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

            # 梯度探针: 必须在主 backward 之前，借助 retain_graph 不破坏主计算图
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pct = train_samples / train_total * 100
            sys.stdout.write(
                f'\r  Epoch {epoch} 训练: {train_samples}/{train_total} ({pct:.1f}%) '
                f'total={metrics["total_loss"]:.4f} denoise={metrics["denoise_loss"]:.4f} t60={metrics["t60_loss"]:.4f}'
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

                with torch.amp.autocast('cuda'):
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

        # 学习率调度按 T60 子损失走 —— 我们关心的是 T60 性能，
        # 单纯按 total 调度会被 alpha/beta 权重歪曲。
        scheduler.step(avg_val_metrics.get('t60_loss', avg_val_loss))
        current_lr = optimizer.param_groups[0]['lr']

        # 早停同样基于 T60 val loss
        early_stop_score = avg_val_metrics.get('t60_loss', avg_val_loss)
        early_stop(early_stop_score, model)
        is_best = early_stop.counter == 0 and early_stop.best_score == early_stop_score
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
            print(f'\n早停! 最佳 T60 验证损失: {early_stop.best_score:.6f}')
            break

    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(early_stop.save_path)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(ckpt['model_state_dict'])
    print(f'已加载最佳模型: {early_stop.save_path}')

    if use_swanlab:
        swanlab_run.finish()

    return model


# ─── 入口 ────────────────────────────────────────────────────────

def load_config_defaults(config_path):
    if not config_path:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError('使用 --config 需要安装 PyYAML: pip install PyYAML') from exc

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    defaults = {}

    def section(name):
        value = cfg.get(name, {})
        return value if isinstance(value, dict) else {}

    data_cfg = section('data')
    for key in ['dataset_root', 'n_fft', 'hop_length', 'audio_length',
                'target_sr', 't60_min', 't60_max', 'log_scale']:
        if key in data_cfg:
            defaults[key] = data_cfg[key]

    train_cfg = section('train')
    for key in [
        'batch_size', 'max_epochs', 'lr', 'weight_decay',
        'early_stop_patience', 'num_workers', 'accum_steps', 'clip_grad_norm',
    ]:
        if key in train_cfg:
            defaults[key] = train_cfg[key]

    test_cfg = section('test')
    if 'batch_size' in test_cfg:
        defaults['test_batch_size'] = test_cfg['batch_size']
    if 'num_workers' in test_cfg:
        defaults['test_num_workers'] = test_cfg['num_workers']
    if 'gpu_id' in test_cfg:
        defaults['test_gpu_id'] = test_cfg['gpu_id']

    model_cfg = section('model')
    if 'n_tscb' in model_cfg:
        defaults['n_tscb'] = model_cfg['n_tscb']
    for key in [
        't60_head_type', 't60_hidden_dim', 't60_dropout',
        't60_fourier_proj_dim', 't60_fourier_first_num_frequencies',
        't60_fourier_hidden_num_frequencies', 't60_fourier_dropout',
        't60_out_activation',
    ]:
        if key in model_cfg:
            defaults[key] = model_cfg[key]
    if 't60_fourier_hidden_dims' in model_cfg:
        defaults['t60_fourier_hidden_dims'] = tuple(model_cfg['t60_fourier_hidden_dims'])

    loss_cfg = section('loss')
    if 'name' in loss_cfg:
        defaults['loss'] = loss_cfg['name']
    if 'huber_delta' in loss_cfg:
        defaults['huber_delta'] = loss_cfg['huber_delta']
    for key in ['alpha', 'beta', 'w_ri', 'w_mag', 'w_time']:
        if key in loss_cfg:
            defaults[key] = loss_cfg[key]

    probe_cfg = section('probe')
    if 'log_every' in probe_cfg:
        defaults['probe_log_every'] = probe_cfg['log_every']
    if 'include_shared_all' in probe_cfg:
        defaults['probe_include_shared_all'] = probe_cfg['include_shared_all']

    exp_cfg = section('experiment')
    exp_name = exp_cfg.get('name')
    output_root = exp_cfg.get('output_root')
    if exp_name:
        defaults['experiment_name'] = exp_name
    if 'save_dir' in exp_cfg:
        defaults['save_dir'] = exp_cfg['save_dir']
    elif output_root and exp_name:
        defaults['save_dir'] = os.path.join(output_root, exp_name)
    elif output_root:
        defaults['save_dir'] = output_root
    if 'save_dir' in test_cfg:
        defaults['test_save_dir'] = test_cfg['save_dir']
    elif output_root and exp_name:
        defaults['test_save_dir'] = os.path.join(output_root, exp_name, 'test')

    logging_cfg = section('logging')
    if 'swanlab_project' in logging_cfg:
        defaults['swanlab_project'] = logging_cfg['swanlab_project']

    return defaults


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description='CMGAN KAN-head 多任务训练 (T60 + 去噪)')
    parser.add_argument('--config', type=str, default=pre_args.config,
                        help='YAML实验配置文件路径')

    # 数据
    parser.add_argument('--dataset_root', type=str,
                        default=config_defaults.get('dataset_root', DEFAULT_DATASET_ROOT))
    parser.add_argument('--n_fft', type=int, default=config_defaults.get('n_fft', 400))
    parser.add_argument('--hop_length', type=int, default=config_defaults.get('hop_length', 100))
    parser.add_argument('--audio_length', type=float, default=config_defaults.get('audio_length', 4.0))
    parser.add_argument('--target_sr', type=int, default=config_defaults.get('target_sr', 16000))
    parser.add_argument('--t60_min', type=float, default=config_defaults.get('t60_min', 0.1))
    parser.add_argument('--t60_max', type=float, default=config_defaults.get('t60_max', 1.5))
    parser.add_argument('--log_scale', action='store_true',
                        default=config_defaults.get('log_scale', False))

    # 训练
    parser.add_argument('--batch_size', type=int, default=config_defaults.get('batch_size', 8))
    parser.add_argument('--max_epochs', type=int, default=config_defaults.get('max_epochs', 100))
    parser.add_argument('--lr', type=float, default=config_defaults.get('lr', 5e-4))
    parser.add_argument('--weight_decay', type=float, default=config_defaults.get('weight_decay', 1e-5))
    parser.add_argument('--early_stop_patience', type=int,
                        default=config_defaults.get('early_stop_patience', 15))
    parser.add_argument('--num_workers', type=int, default=config_defaults.get('num_workers', 4))
    parser.add_argument('--accum_steps', type=int, default=config_defaults.get('accum_steps', 2))
    parser.add_argument('--clip_grad_norm', type=float, default=config_defaults.get('clip_grad_norm', 5.0))
    parser.add_argument('--n_tscb', type=int, default=config_defaults.get('n_tscb', 2),
                        choices=[2, 4], help='TSCB层数')

    # T60 head
    parser.add_argument('--t60_head_type', type=str,
                        default=config_defaults.get('t60_head_type', 'fourier_kan'),
                        choices=['mlp', 'fourier_kan'])
    parser.add_argument('--t60_hidden_dim', type=int,
                        default=config_defaults.get('t60_hidden_dim', 128))
    parser.add_argument('--t60_dropout', type=float,
                        default=config_defaults.get('t60_dropout', 0.3))
    parser.add_argument('--t60_fourier_proj_dim', type=int,
                        default=config_defaults.get('t60_fourier_proj_dim', 64))
    parser.add_argument('--t60_fourier_first_num_frequencies', type=int,
                        default=config_defaults.get('t60_fourier_first_num_frequencies', 16))
    parser.add_argument('--t60_fourier_hidden_num_frequencies', type=int,
                        default=config_defaults.get('t60_fourier_hidden_num_frequencies', 8))
    parser.add_argument('--t60_fourier_dropout', type=float,
                        default=config_defaults.get('t60_fourier_dropout', 0.1))
    parser.add_argument('--t60_out_activation', type=str,
                        default=config_defaults.get('t60_out_activation', 'sigmoid'),
                        choices=['sigmoid', 'softplus', 'identity', 'none'])

    # 损失
    parser.add_argument('--loss', type=str, default=config_defaults.get('loss', 'mse'),
                        choices=['mse', 'mae', 'huber'], help='T60子损失类型')
    parser.add_argument('--huber_delta', type=float, default=config_defaults.get('huber_delta', 1.0))
    parser.add_argument('--alpha', type=float, default=config_defaults.get('alpha', 1.0),
                        help='去噪损失权重')
    parser.add_argument('--beta', type=float, default=config_defaults.get('beta', 1.0),
                        help='T60损失权重')
    parser.add_argument('--w_ri', type=float, default=config_defaults.get('w_ri', 0.1))
    parser.add_argument('--w_mag', type=float, default=config_defaults.get('w_mag', 0.9))
    parser.add_argument('--w_time', type=float, default=config_defaults.get('w_time', 0.2))

    # 梯度探针
    parser.add_argument('--probe_log_every', type=int,
                        default=config_defaults.get('probe_log_every', 20))
    parser.add_argument('--probe_include_shared_all', action='store_true',
                        default=config_defaults.get('probe_include_shared_all', False))

    # 输出
    parser.add_argument('--save_dir', type=str, default=config_defaults.get('save_dir', 'runs'))
    parser.add_argument('--experiment_name', type=str,
                        default=config_defaults.get('experiment_name', 'CMGAN_kan_multitask'))
    parser.add_argument('--swanlab_project', type=str,
                        default=config_defaults.get('swanlab_project', 'T60_KAN_MultiTask'))

    # 测试 (训练后会沿用单任务 test.py 自动评估 T60；本任务不评估去噪)
    parser.add_argument('--test_batch_size', type=int,
                        default=config_defaults.get('test_batch_size',
                                                    config_defaults.get('batch_size', 8)))
    parser.add_argument('--test_num_workers', type=int,
                        default=config_defaults.get('test_num_workers',
                                                    config_defaults.get('num_workers', 4)))
    parser.add_argument('--test_save_dir', type=str,
                        default=config_defaults.get('test_save_dir', None))
    parser.add_argument('--test_gpu_id', type=int, default=config_defaults.get('test_gpu_id', 0))

    # 模式
    parser.add_argument('--gpu_id', type=int, default=0)
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

        test_args = argparse.Namespace(**vars(args))
        test_args.model_path = os.path.join(args.save_dir, 'best_model.pth')
        test_args.batch_size = args.test_batch_size
        test_args.num_workers = args.test_num_workers
        test_args.save_dir = args.test_save_dir or os.path.join(args.save_dir, 'test')
        test_args.gpu_id = args.test_gpu_id
        args.model_path = test_args.model_path

    if args.model_path:
        # 复用单任务 test.run_tests，但需要它知道多任务模型的 forward 返回三元组。
        # 这里用 test_multitask.run_tests，单独实现以避免修改原 test.py。
        from test_multitask import run_tests
        run_tests(test_args if not args.test_only else args)
