"""
CMGAN 单任务训练: T60估计
- 只保留 CMGAN encoder/TSCB backbone + T60 head
- SwanLab 实验追踪
- ReduceLROnPlateau + Early Stopping
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generator_t60 import TSCNet_T60Estimator, TSCNet_T60Estimator_2TSCB
from dataset import DEFAULT_DATASET_ROOT, create_dataloaders, resolve_dataset_root
from utils import power_compress


# ─── T60 单任务损失 ──────────────────────────────────────────────

class T60Loss(nn.Module):
    """T60 regression loss on normalized labels."""

    def __init__(self, loss_type='mse', huber_delta=1.0):
        super().__init__()
        if loss_type not in {'mse', 'mae', 'huber'}:
            raise ValueError(f'Unsupported loss_type: {loss_type}')
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(self, t60_pred, t60_target):
        if self.loss_type == 'mse':
            loss = F.mse_loss(t60_pred, t60_target)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(t60_pred, t60_target)
        else:
            loss = F.huber_loss(t60_pred, t60_target, delta=self.huber_delta)

        return loss, {
            't60_loss': loss.item(),
            'total_loss': loss.item(),
        }


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
        # 兼容 DataParallel: 取原始模型
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({'model_state_dict': raw_model.state_dict(), 'best_score': self.best_score}, self.save_path)


# ─── 主训练逻辑 ──────────────────────────────────────────────────

def train(args):
    args.dataset_root = resolve_dataset_root(args.dataset_root)
    if args.config:
        print(f'配置文件: {args.config}')
    print(f'数据集路径: {args.dataset_root}')
    print(f'固定音频长度: {args.audio_length:.2f}s ({int(args.audio_length * args.target_sr)} samples)')

    # 设备 & 多 GPU
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0')
    print(f'可用 GPU: {n_gpu} 张 ({[torch.cuda.get_device_name(i) for i in range(n_gpu)]})')

    # SwanLab
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

    # 数据
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

    # fourier_hidden_dims: 可能来自 YAML（tuple）或 argparse（未暴露，使用默认值）
    fourier_hidden_dims = getattr(args, 't60_fourier_hidden_dims', (32, 16))

    # 模型
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
        model = TSCNet_T60Estimator_2TSCB(**model_kwargs)
    else:
        model = TSCNet_T60Estimator(**model_kwargs)
    model = model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
        print(f'使用 DataParallel × {n_gpu} GPU')
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_params = sum(p.numel() for p in raw_model.parameters())
    head_params = sum(p.numel() for p in raw_model.t60_head.parameters())
    print(f'T60 head 类型: {args.t60_head_type}')
    print(f'T60 head 参数: {head_params:,}')
    print(f'模型总参数: {raw_params:,} ({raw_params / 1e6:.2f}M)')

    # 损失
    criterion = T60Loss(loss_type=args.loss, huber_delta=args.huber_delta)
    print(f'T60 loss: {args.loss}')

    # 优化器 + AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 早停
    os.makedirs(args.save_dir, exist_ok=True)
    early_stop = EarlyStopping(
        patience=args.early_stop_patience, mode='min',
        save_path=os.path.join(args.save_dir, 'best_model.pth'),
    )

    # 训练循环
    history = []
    print(f'\n{"Epoch":>6} {"Train Loss":>10} {"Val Loss":>10} {"T60":>10} {"LR":>12}')
    print('─' * 54)

    # 梯度累积步数 (有效 batch = batch_size × accum_steps)
    accum_steps = args.accum_steps
    effective_batch = args.batch_size * accum_steps
    print(f'实际 batch={args.batch_size}, 累积 {accum_steps} 步, 等效 batch={effective_batch}')

    train_total = len(train_loader.dataset)
    val_total = len(eval_loader.dataset)

    for epoch in range(1, args.max_epochs + 1):
        # ── 训练 ──
        model.train()
        train_loss_sum = 0
        train_metrics_sum = {}
        train_count = 0
        train_samples = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            noisy_spec = batch['noisy_spec'].to(device)  # (B, 2, T, F) 原始STFT
            t60_target = batch['t60'].to(device)

            # 功率压缩: (B, 2, T, F) → (B, F, T, 2) → power_compress → (B, 2, F, T) → permute → (B, 2, T, F)
            noisy_input = power_compress(noisy_spec.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)

            # AMP 混合精度前向
            with torch.amp.autocast('cuda'):
                t60_pred = model(noisy_input)
                loss, metrics = criterion(t60_pred, t60_target)
                # 梯度累积: 缩放损失
                scaled_loss = loss / accum_steps

            scaler.scale(scaled_loss).backward()

            train_loss_sum += metrics['total_loss']
            for k, v in metrics.items():
                train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
            train_count += 1
            train_samples += noisy_spec.size(0)

            # 累积完成后才更新参数
            if (step_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 进度条
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
                t60_target = batch['t60'].to(device)

                noisy_input = power_compress(noisy_spec.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)

                with torch.cuda.amp.autocast():
                    t60_pred = model(noisy_input)
                    loss, metrics = criterion(t60_pred, t60_target)

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

        # 学习率调度
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 早停
        early_stop(avg_val_loss, model)
        is_best = early_stop.counter == 0 and early_stop.best_score == avg_val_loss
        marker = ' *' if is_best else ''

        # 打印
        print(f'{epoch:>6d} {avg_train_loss:>10.4f} {avg_val_loss:>10.4f} '
              f'{avg_val_metrics.get("t60_loss", 0):>10.4f} '
              f'{current_lr:>12.2e}{marker}')

        # SwanLab
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
            swanlab_run.log(log_dict, step=epoch)

        # 历史
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

    # 保存训练历史
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 加载最佳模型
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
    for key in ['dataset_root', 'n_fft', 'hop_length', 'audio_length', 'target_sr', 't60_min', 't60_max']:
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
        't60_fourier_hidden_num_frequencies', 't60_fourier_dropout', 't60_out_activation',
    ]:
        if key in model_cfg:
            defaults[key] = model_cfg[key]
    # fourier_hidden_dims 是列表类型，单独处理
    if 't60_fourier_hidden_dims' in model_cfg:
        defaults['t60_fourier_hidden_dims'] = tuple(model_cfg['t60_fourier_hidden_dims'])

    loss_cfg = section('loss')
    if 'name' in loss_cfg:
        defaults['loss'] = loss_cfg['name']
    if 'huber_delta' in loss_cfg:
        defaults['huber_delta'] = loss_cfg['huber_delta']

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

    parser = argparse.ArgumentParser(description='CMGAN 单任务训练: T60估计')
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

    # 训练
    parser.add_argument('--batch_size', type=int, default=config_defaults.get('batch_size', 8))
    parser.add_argument('--max_epochs', type=int, default=config_defaults.get('max_epochs', 100))
    parser.add_argument('--lr', type=float, default=config_defaults.get('lr', 5e-4))
    parser.add_argument('--weight_decay', type=float, default=config_defaults.get('weight_decay', 1e-5))
    parser.add_argument('--early_stop_patience', type=int, default=config_defaults.get('early_stop_patience', 15))
    parser.add_argument('--num_workers', type=int, default=config_defaults.get('num_workers', 4))
    parser.add_argument('--accum_steps', type=int, default=config_defaults.get('accum_steps', 2), help='梯度累积步数')
    parser.add_argument('--clip_grad_norm', type=float, default=config_defaults.get('clip_grad_norm', 5.0), help='梯度裁剪 max_norm')
    parser.add_argument('--n_tscb', type=int, default=config_defaults.get('n_tscb', 4), choices=[2, 4], help='TSCB层数')

    # T60 head 类型与超参数
    parser.add_argument(
        '--t60_head_type', type=str,
        default=config_defaults.get('t60_head_type', 'mlp'),
        choices=['mlp', 'fourier_kan'],
        help='T60回归头类型: mlp（默认）或 fourier_kan',
    )
    parser.add_argument('--t60_hidden_dim', type=int,
                        default=config_defaults.get('t60_hidden_dim', 128),
                        help='MLP head 隐层宽度（仅 t60_head_type=mlp 有效）')
    parser.add_argument('--t60_dropout', type=float,
                        default=config_defaults.get('t60_dropout', 0.3),
                        help='MLP head Dropout（仅 t60_head_type=mlp 有效）')
    parser.add_argument('--t60_fourier_proj_dim', type=int,
                        default=config_defaults.get('t60_fourier_proj_dim', 64),
                        help='Fourier-KAN head 投影维度')
    parser.add_argument('--t60_fourier_first_num_frequencies', type=int,
                        default=config_defaults.get('t60_fourier_first_num_frequencies', 16),
                        help='Fourier-KAN head 第一层 Omega（较大值）')
    parser.add_argument('--t60_fourier_hidden_num_frequencies', type=int,
                        default=config_defaults.get('t60_fourier_hidden_num_frequencies', 8),
                        help='Fourier-KAN head 后续隐层 Omega（较小值）')
    parser.add_argument('--t60_fourier_dropout', type=float,
                        default=config_defaults.get('t60_fourier_dropout', 0.1),
                        help='Fourier-KAN head Dropout')
    parser.add_argument(
        '--t60_out_activation', type=str,
        default=config_defaults.get('t60_out_activation', 'sigmoid'),
        choices=['sigmoid', 'softplus', 'identity', 'none'],
        help='Fourier-KAN head 输出激活函数',
    )

    # 损失
    parser.add_argument('--loss', type=str, default=config_defaults.get('loss', 'mse'),
                        choices=['mse', 'mae', 'huber'], help='T60单任务损失')
    parser.add_argument('--huber_delta', type=float, default=config_defaults.get('huber_delta', 1.0))

    # 输出
    parser.add_argument('--save_dir', type=str, default=config_defaults.get('save_dir', 'runs'))
    parser.add_argument('--experiment_name', type=str, default=config_defaults.get('experiment_name', 'CMGAN_single_t60'))
    parser.add_argument('--swanlab_project', type=str, default=config_defaults.get('swanlab_project', 'T60_Estimation'))

    # 测试配置。train-and-test.sh 会在训练结束后沿用这些参数自动评估 test1-test4。
    parser.add_argument('--test_batch_size', type=int,
                        default=config_defaults.get('test_batch_size', config_defaults.get('batch_size', 8)))
    parser.add_argument('--test_num_workers', type=int,
                        default=config_defaults.get('test_num_workers', config_defaults.get('num_workers', 4)))
    parser.add_argument('--test_save_dir', type=str, default=config_defaults.get('test_save_dir', None))
    parser.add_argument('--test_gpu_id', type=int, default=config_defaults.get('test_gpu_id', 0))

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

        # 训练后自动测试
        test_args = argparse.Namespace(**vars(args))
        test_args.model_path = os.path.join(args.save_dir, 'best_model.pth')
        test_args.batch_size = args.test_batch_size
        test_args.num_workers = args.test_num_workers
        test_args.save_dir = args.test_save_dir or os.path.join(args.save_dir, 'test')
        test_args.gpu_id = args.test_gpu_id
        args.model_path = test_args.model_path

    if args.model_path:
        from test import run_tests
        run_tests(test_args if not args.test_only else args)
