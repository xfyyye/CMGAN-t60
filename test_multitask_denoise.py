"""
KAN 多任务模型的去噪性能测试 (test1-test4)。

评估指标:
  - PESQ  (wb): ITU-T P.862.2 宽带感知语音质量，[-0.5, 4.5]，越高越好
  - STOI      : 短时客观可懂度，[0, 1]，越高越好
  - SI-SDR    : 尺度不变信噪比 (dB)，越高越好

同时输出 noisy 基线指标和各指标的提升量 (improvement)。

去噪 target 是数据集中的 *_denoised.wav（纯混响、无噪声），
即任务是去噪而非去混响，增强后的语音仍保留混响。

Usage:
    # 用 config 自动读取模型参数（推荐）
    CUDA_VISIBLE_DEVICES=5 python test_multitask_denoise.py \\
        --config configs/t60_multitask/kan_mse_w_denoise.yaml \\
        --model_path runs/kan_multitask_w_denoise/best_model.pth

    # 手动指定所有参数
    CUDA_VISIBLE_DEVICES=5 python test_multitask_denoise.py \\
        --model_path runs/kan_multitask_w_denoise/best_model.pth \\
        --dataset_root /path/to/T60_Dataset_v7_4s \\
        --n_tscb 2 --gpu_id 0

    # 同时保存增强后的音频（每测试集最多 5 条）
    CUDA_VISIBLE_DEVICES=5 python test_multitask_denoise.py \\
        --config configs/t60_multitask/kan_mse_w_denoise.yaml \\
        --model_path runs/kan_multitask_w_denoise/best_model.pth \\
        --save_wav --max_wav_save 5

结果保存到 runs/{experiment_name}/test_denoise/denoise_evaluation.json。
"""

import os
import json
import argparse

import numpy as np
import torch
import torchaudio

from models.generator_t60 import (
    TSCNet_KAN_MultiTask,
    TSCNet_KAN_MultiTask_2TSCB,
)
from dataset import (
    DEFAULT_DATASET_ROOT,
    resolve_dataset_root,
    CMGANT60MultiTaskDataset,
    collate_fn_multitask,
)
from torch.utils.data import DataLoader
from utils import power_compress, power_uncompress


# ─── 指标 ─────────────────────────────────────────────────────────────────────

def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8)))


# ─── 主测试逻辑 ───────────────────────────────────────────────────────────────

def run_denoise_test(args):
    from pesq import pesq as calc_pesq
    from pystoi import stoi as calc_stoi

    args.dataset_root = resolve_dataset_root(args.dataset_root)
    print(f'数据集路径: {args.dataset_root}')

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 加载 KAN 多任务模型
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
    model_cls = TSCNet_KAN_MultiTask_2TSCB if args.n_tscb == 2 else TSCNet_KAN_MultiTask
    model = model_cls(**model_kwargs)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model = model.to(device).eval()
    print(f'T60 head 类型: {args.t60_head_type}')
    print(f'已加载模型: {args.model_path}')

    os.makedirs(args.save_dir, exist_ok=True)
    wav_dir = os.path.join(args.save_dir, 'denoised_wavs')
    if args.save_wav:
        os.makedirs(wav_dir, exist_ok=True)

    n_fft = args.n_fft
    hop   = args.hop_length
    window = torch.hamming_window(n_fft)

    summary = {}

    for test_split in ['test1', 'test2', 'test3', 'test4']:
        print(f'\n{"=" * 60}')
        print(f'评估去噪 {test_split}')
        print(f'{"=" * 60}')

        ds = CMGANT60MultiTaskDataset(
            dataset_root=args.dataset_root,
            split=test_split,
            n_fft=n_fft,
            hop_length=hop,
            audio_length=args.audio_length,
            target_sr=args.target_sr,
        )
        loader = DataLoader(
            ds, batch_size=1, shuffle=False,
            num_workers=2, collate_fn=collate_fn_multitask,
        )

        all_pesq,       all_stoi,       all_sisdr       = [], [], []
        all_pesq_noisy, all_stoi_noisy, all_sisdr_noisy = [], [], []

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                noisy_spec  = batch['noisy_spec'].to(device)
                noisy_wav   = batch['noisy_wav'].numpy()[0]
                clean_wav   = batch['clean_wav'].numpy()[0]
                sample_name = batch['sample_name'][0]

                # 能量归一化系数（用于还原幅度）
                c = np.sqrt(noisy_wav.size / (np.sum(noisy_wav ** 2) + 1e-8))

                # 前向推理
                noisy_input = power_compress(
                    noisy_spec.permute(0, 3, 2, 1)
                ).permute(0, 1, 3, 2)

                with torch.amp.autocast('cuda'):
                    est_real, est_imag, _ = model(noisy_input)

                # 解压 + ISTFT：模型输出 (B,1,T,F)，permute 后 (B,1,F,T) 再 uncompress
                est_real_f = est_real.permute(0, 1, 3, 2)
                est_imag_f = est_imag.permute(0, 1, 3, 2)
                est_spec_uncompress = power_uncompress(est_real_f, est_imag_f).squeeze(1)
                est_spec_complex = torch.complex(
                    est_spec_uncompress[..., 0], est_spec_uncompress[..., 1]
                )
                est_audio = torch.istft(
                    est_spec_complex, n_fft, hop,
                    window=window.to(device), onesided=True,
                )
                est_wav = est_audio.cpu().numpy()[0]

                # 对齐长度
                min_len  = min(len(est_wav), len(clean_wav), len(noisy_wav))
                est_wav   = est_wav[:min_len]
                clean_wav = clean_wav[:min_len]
                noisy_wav = noisy_wav[:min_len]

                # 还原能量归一化
                est_wav   = est_wav   / (c + 1e-8)
                clean_wav = clean_wav / (c + 1e-8)
                noisy_wav = noisy_wav / (c + 1e-8)

                # 增强后指标
                try:
                    p = calc_pesq(args.target_sr, clean_wav, est_wav, 'wb')
                except Exception:
                    p = float('nan')
                s = calc_stoi(clean_wav, est_wav, args.target_sr)
                d = si_sdr(est_wav, clean_wav)

                # noisy 基线指标
                try:
                    p_noisy = calc_pesq(args.target_sr, clean_wav, noisy_wav, 'wb')
                except Exception:
                    p_noisy = float('nan')
                s_noisy = calc_stoi(clean_wav, noisy_wav, args.target_sr)
                d_noisy = si_sdr(noisy_wav, clean_wav)

                all_pesq.append(p);             all_stoi.append(s);             all_sisdr.append(d)
                all_pesq_noisy.append(p_noisy); all_stoi_noisy.append(s_noisy); all_sisdr_noisy.append(d_noisy)

                # 保存音频
                if args.save_wav and idx < args.max_wav_save:
                    for tag, wav in [('enhanced', est_wav), ('noisy', noisy_wav), ('clean', clean_wav)]:
                        torchaudio.save(
                            os.path.join(wav_dir, f'{test_split}_{sample_name}_{tag}.wav'),
                            torch.FloatTensor(wav).unsqueeze(0), args.target_sr,
                        )

                if (idx + 1) % 100 == 0:
                    print(f'  {idx + 1}/{len(ds)} 样本已处理')

        avg_pesq       = np.nanmean(all_pesq)
        avg_stoi       = np.nanmean(all_stoi)
        avg_sisdr      = np.nanmean(all_sisdr)
        avg_pesq_noisy = np.nanmean(all_pesq_noisy)
        avg_stoi_noisy = np.nanmean(all_stoi_noisy)
        avg_sisdr_noisy = np.nanmean(all_sisdr_noisy)

        result = {
            'test_split':  test_split,
            'num_samples': len(ds),
            'enhanced': {
                'PESQ':     round(avg_pesq,  4),
                'STOI':     round(avg_stoi,  4),
                'SI_SDR_dB': round(avg_sisdr, 2),
            },
            'noisy_input': {
                'PESQ':     round(avg_pesq_noisy,  4),
                'STOI':     round(avg_stoi_noisy,  4),
                'SI_SDR_dB': round(avg_sisdr_noisy, 2),
            },
            'improvement': {
                'PESQ':     round(avg_pesq  - avg_pesq_noisy,  4),
                'STOI':     round(avg_stoi  - avg_stoi_noisy,  4),
                'SI_SDR_dB': round(avg_sisdr - avg_sisdr_noisy, 2),
            },
        }
        summary[test_split] = result

        print(f'  增强:  PESQ={avg_pesq:.3f} | STOI={avg_stoi:.3f} | SI-SDR={avg_sisdr:.2f} dB')
        print(f'  输入:  PESQ={avg_pesq_noisy:.3f} | STOI={avg_stoi_noisy:.3f} | SI-SDR={avg_sisdr_noisy:.2f} dB')
        print(f'  提升:  PESQ={avg_pesq-avg_pesq_noisy:+.3f} | STOI={avg_stoi-avg_stoi_noisy:+.3f} | SI-SDR={avg_sisdr-avg_sisdr_noisy:+.2f} dB')

    # 四测试集均值
    splits = ['test1', 'test2', 'test3', 'test4']
    print(f'\n{"=" * 60}')
    print('四测试集平均')
    print(f'{"=" * 60}')
    avg = {
        cat: {
            metric: round(np.mean([summary[t][cat][metric] for t in splits]), 4 if metric != 'SI_SDR_dB' else 2)
            for metric in ['PESQ', 'STOI', 'SI_SDR_dB']
        }
        for cat in ['enhanced', 'noisy_input', 'improvement']
    }
    print(f'  增强:  PESQ={avg["enhanced"]["PESQ"]:.3f} | STOI={avg["enhanced"]["STOI"]:.3f} | SI-SDR={avg["enhanced"]["SI_SDR_dB"]:.2f} dB')
    print(f'  输入:  PESQ={avg["noisy_input"]["PESQ"]:.3f} | STOI={avg["noisy_input"]["STOI"]:.3f} | SI-SDR={avg["noisy_input"]["SI_SDR_dB"]:.2f} dB')
    print(f'  提升:  PESQ={avg["improvement"]["PESQ"]:+.3f} | STOI={avg["improvement"]["STOI"]:+.3f} | SI-SDR={avg["improvement"]["SI_SDR_dB"]:+.2f} dB')

    out_path = os.path.join(args.save_dir, 'denoise_evaluation.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'average': avg}, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')
    return summary


# ─── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args():
    from test import load_config_defaults

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description='KAN 多任务去噪性能测试')
    parser.add_argument('--config',      type=str, default=pre_args.config)
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--dataset_root', type=str,
                        default=config_defaults.get('dataset_root', DEFAULT_DATASET_ROOT))
    parser.add_argument('--n_fft',       type=int,   default=config_defaults.get('n_fft', 400))
    parser.add_argument('--hop_length',  type=int,   default=config_defaults.get('hop_length', 100))
    parser.add_argument('--audio_length', type=float, default=config_defaults.get('audio_length', 4.0))
    parser.add_argument('--target_sr',   type=int,   default=config_defaults.get('target_sr', 16000))
    parser.add_argument('--n_tscb',      type=int,   default=config_defaults.get('n_tscb', 2),
                        choices=[2, 4])

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

    parser.add_argument('--save_dir',    type=str,   default='test_results_denoise')
    parser.add_argument('--experiment_name', type=str,
                        default=config_defaults.get('experiment_name', 'CMGAN_kan_multitask'))
    parser.add_argument('--gpu_id',      type=int,   default=0)
    parser.add_argument('--save_wav',    action='store_true', help='保存增强后的音频文件')
    parser.add_argument('--max_wav_save', type=int,  default=5, help='每测试集最多保存的音频数')
    return parser.parse_args()


if __name__ == '__main__':
    run_denoise_test(parse_args())
