"""
CMGAN T60 单任务评估脚本
对 test1-test4 全部运行，输出标准T60指标
JSON 格式与 /compare skill 的 Format B 兼容
"""

import os
import json
import argparse
import numpy as np
import torch
from scipy import stats

from models.generator_t60 import TSCNet_T60Estimator, TSCNet_T60Estimator_2TSCB
from dataset import DEFAULT_DATASET_ROOT, create_test_loader, resolve_dataset_root, T60Normalizer
from utils import power_compress


# ─── 标准分箱 ──────────────────────────────────────────────────────

T60_BIN_EDGES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
T60_BIN_LABELS = [f"T60_{T60_BIN_EDGES[i]}-{T60_BIN_EDGES[i+1]}s"
                  for i in range(len(T60_BIN_EDGES) - 1)]
SNR_VALUES = [-5, 0, 5, 10, 15, 20]
SNR_LABELS = [f"SNR_{snr}dB" for snr in SNR_VALUES]


# ─── 指标计算 ──────────────────────────────────────────────────────

def compute_standard_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = y_pred - y_true
    bias_ms = float(np.mean(errors) * 1000)
    rmse_ms = float(np.sqrt(np.mean(errors ** 2)) * 1000)
    mae_ms = float(np.mean(np.abs(errors)) * 1000)
    pearson_r = float(stats.pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float('nan')
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    mean_rel = float(np.mean(np.abs(errors) / np.maximum(np.abs(y_true), 1e-8)) * 100)
    return {
        'rmse_ms': round(rmse_ms, 2), 'mae_ms': round(mae_ms, 2),
        'bias_ms': round(bias_ms, 2), 'pearson_r': round(pearson_r, 4),
        'r2_score': round(r2_score, 4), 'mean_rel_error_pct': round(mean_rel, 2),
    }


def compute_grouped_metrics(y_true, y_pred, t60_raw, snr_values):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    t60_raw, snr_arr = np.array(t60_raw), np.array(snr_values)
    result = {}

    # T60 分组
    bin_analysis = {}
    for i in range(len(T60_BIN_EDGES) - 1):
        low, high = T60_BIN_EDGES[i], T60_BIN_EDGES[i + 1]
        mask = (t60_raw >= low) & (t60_raw < high)
        count = int(mask.sum())
        if count > 0:
            m = compute_standard_metrics(y_true[mask], y_pred[mask])
            bin_analysis[T60_BIN_LABELS[i]] = {
                'count': count, 'bias_ms': m['bias_ms'],
                'rmse_ms': m['rmse_ms'], 'mae_ms': m['mae_ms'],
                'pearson_r': m['pearson_r'],
            }
        else:
            bin_analysis[T60_BIN_LABELS[i]] = {
                'count': 0, 'bias_ms': float('nan'),
                'rmse_ms': float('nan'), 'mae_ms': float('nan'),
                'pearson_r': float('nan'),
            }
    result['bin_analysis'] = bin_analysis

    # SNR 分组
    snr_analysis = {}
    for i, snr in enumerate(SNR_VALUES):
        mask = snr_arr == snr
        count = int(mask.sum())
        if count > 0:
            m = compute_standard_metrics(y_true[mask], y_pred[mask])
            snr_analysis[SNR_LABELS[i]] = {
                'count': count, 'bias_ms': m['bias_ms'],
                'rmse_ms': m['rmse_ms'], 'mae_ms': m['mae_ms'],
                'pearson_r': m['pearson_r'],
            }
        else:
            snr_analysis[SNR_LABELS[i]] = {
                'count': 0, 'bias_ms': float('nan'),
                'rmse_ms': float('nan'), 'mae_ms': float('nan'),
                'pearson_r': float('nan'),
            }
    result['snr_analysis'] = snr_analysis

    return result


# ─── 测试主逻辑 ──────────────────────────────────────────────────

def run_tests(args):
    args.dataset_root = resolve_dataset_root(args.dataset_root)
    print(f'数据集路径: {args.dataset_root}')

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 加载模型
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
        model = TSCNet_T60Estimator_2TSCB(**model_kwargs)
    else:
        model = TSCNet_T60Estimator(**model_kwargs)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model = model.to(device)
    model.eval()
    print(f'T60 head 类型: {args.t60_head_type}')
    print(f'已加载模型: {args.model_path}')

    t60_normalizer = T60Normalizer(args.t60_min, args.t60_max)
    os.makedirs(args.save_dir, exist_ok=True)

    summary = {}

    for test_split in ['test1', 'test2', 'test3', 'test4']:
        print(f'\n{"=" * 60}')
        print(f'评估 {test_split}')
        print(f'{"=" * 60}')

        loader = create_test_loader(
            test_split=test_split,
            dataset_root=args.dataset_root,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            audio_length=args.audio_length,
            target_sr=args.target_sr,
            t60_range=(args.t60_min, args.t60_max),
        )

        all_true, all_pred, all_names, all_snrs = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                noisy_spec = batch['noisy_spec'].to(device)
                t60_raw = batch['t60_raw'].numpy()

                noisy_input = power_compress(noisy_spec.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)

                with torch.cuda.amp.autocast():
                    t60_pred = model(noisy_input)

                pred_t60 = t60_pred.cpu().numpy()
                pred_t60 = t60_normalizer.denormalize(pred_t60)

                all_true.extend(t60_raw.tolist())
                all_pred.extend(pred_t60.tolist())
                all_snrs.extend(batch['snr'].numpy().tolist())
                all_names.extend(batch['sample_name'])

        # 计算指标
        standard = compute_standard_metrics(all_true, all_pred)
        grouped = compute_grouped_metrics(all_true, all_pred, all_true, all_snrs)

        results = {
            'experiment_name': args.experiment_name,
            'dataset_version': 'T60_Dataset_v7',
            'test_split': test_split,
            'standard_metrics': standard,
            'bin_analysis': grouped['bin_analysis'],
            'snr_analysis': grouped['snr_analysis'],
            'per_sample_results': [
                {'sample_name': n, 'true_t60': round(float(t), 4),
                 'pred_t60': round(float(p), 4), 'snr': float(s),
                 'error_ms': round((p - t) * 1000, 2)}
                for n, t, p, s in zip(all_names, all_true, all_pred, all_snrs)
            ],
        }

        # 保存
        out_path = os.path.join(args.save_dir, f'evaluation_results_{test_split}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f'  RMSE: {standard["rmse_ms"]:.2f} ms | MAE: {standard["mae_ms"]:.2f} ms | '
              f'Pearson r: {standard["pearson_r"]:.4f} | R²: {standard["r2_score"]:.4f}')

        summary[test_split] = standard

    # 总体平均
    print(f'\n{"=" * 60}')
    print('四测试集平均')
    print(f'{"=" * 60}')
    avg = {k: np.mean([summary[t][k] for t in ['test1','test2','test3','test4']])
           for k in ['rmse_ms', 'mae_ms', 'pearson_r', 'r2_score']}
    print(f'  RMSE: {avg["rmse_ms"]:.2f} ms | MAE: {avg["mae_ms"]:.2f} ms | '
          f'Pearson r: {avg["pearson_r"]:.4f} | R²: {avg["r2_score"]:.4f}')

    # 保存汇总
    summary_path = os.path.join(args.save_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'summary': summary, 'average': avg}, f, indent=2)

    return summary


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
    if 't60_fourier_hidden_dims' in model_cfg:
        defaults['t60_fourier_hidden_dims'] = tuple(model_cfg['t60_fourier_hidden_dims'])

    test_cfg = section('test')
    for key in ['batch_size', 'num_workers', 'gpu_id']:
        if key in test_cfg:
            defaults[key] = test_cfg[key]

    exp_cfg = section('experiment')
    exp_name = exp_cfg.get('name')
    output_root = exp_cfg.get('output_root')
    if exp_name:
        defaults['experiment_name'] = exp_name
    if 'save_dir' in test_cfg:
        defaults['save_dir'] = test_cfg['save_dir']
    elif output_root and exp_name:
        defaults['save_dir'] = os.path.join(output_root, exp_name, 'test')

    return defaults


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description='CMGAN T60 测试')
    parser.add_argument('--config', type=str, default=pre_args.config,
                        help='YAML实验配置文件路径')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_root', type=str,
                        default=config_defaults.get('dataset_root', DEFAULT_DATASET_ROOT))
    parser.add_argument('--n_fft', type=int, default=config_defaults.get('n_fft', 400))
    parser.add_argument('--hop_length', type=int, default=config_defaults.get('hop_length', 100))
    parser.add_argument('--audio_length', type=float, default=config_defaults.get('audio_length', 4.0))
    parser.add_argument('--target_sr', type=int, default=config_defaults.get('target_sr', 16000))
    parser.add_argument('--t60_min', type=float, default=config_defaults.get('t60_min', 0.1))
    parser.add_argument('--t60_max', type=float, default=config_defaults.get('t60_max', 1.5))
    parser.add_argument('--batch_size', type=int, default=config_defaults.get('batch_size', 8))
    parser.add_argument('--n_tscb', type=int, default=config_defaults.get('n_tscb', 4), choices=[2, 4])

    # T60 head 类型与超参数（与 train.py 保持一致）
    parser.add_argument(
        '--t60_head_type', type=str,
        default=config_defaults.get('t60_head_type', 'mlp'),
        choices=['mlp', 'fourier_kan'],
        help='T60回归头类型: mlp（默认）或 fourier_kan',
    )
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
    parser.add_argument(
        '--t60_out_activation', type=str,
        default=config_defaults.get('t60_out_activation', 'sigmoid'),
        choices=['sigmoid', 'softplus', 'identity', 'none'],
    )

    parser.add_argument('--num_workers', type=int, default=config_defaults.get('num_workers', 4))
    parser.add_argument('--save_dir', type=str, default=config_defaults.get('save_dir', 'test_results'))
    parser.add_argument('--experiment_name', type=str,
                        default=config_defaults.get('experiment_name', 'CMGAN_single_t60'))
    parser.add_argument('--gpu_id', type=int, default=config_defaults.get('gpu_id', 0))
    return parser.parse_args()


if __name__ == '__main__':
    run_tests(parse_args())
