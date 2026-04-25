"""
CMGAN T60 多任务去噪评估脚本
对 test1-test4 运行，计算标准语音增强指标: PESQ, STOI, SI-SDR
同时保存增强后的音频文件供试听
"""

import os
import json
import argparse
import numpy as np
import torch
import torchaudio

from models.generator_t60 import TSCNet_MultiTask, TSCNet_MultiTask_2TSCB
from utils import power_compress, power_uncompress


def si_sdr(est, ref):
    """Scale-Invariant Signal-to-Distortion Ratio (dB)"""
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise = est - target
    val = 10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8))
    return float(val)


def run_denoise_test(args):
    from pesq import pesq as calc_pesq
    from pystoi import stoi as calc_stoi

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    if args.n_tscb == 2:
        model = TSCNet_MultiTask_2TSCB(num_channel=64, num_features=args.n_fft // 2 + 1)
    else:
        model = TSCNet_MultiTask(num_channel=64, num_features=args.n_fft // 2 + 1)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model = model.to(device)
    model.eval()
    print(f'已加载模型: {args.model_path}')

    os.makedirs(args.save_dir, exist_ok=True)
    wav_dir = os.path.join(args.save_dir, 'denoised_wavs')
    if args.save_wav:
        os.makedirs(wav_dir, exist_ok=True)

    n_fft = args.n_fft
    hop = args.hop_length
    window = torch.hamming_window(n_fft)

    summary = {}

    for test_split in ['test1', 'test2', 'test3', 'test4']:
        print(f'\n{"=" * 60}')
        print(f'评估去噪 {test_split}')
        print(f'{"=" * 60}')

        from dataset import CMGANT60Dataset, collate_fn
        from torch.utils.data import DataLoader

        ds = CMGANT60Dataset(
            dataset_root=args.dataset_root, split=test_split,
            n_fft=n_fft, hop_length=hop,
            audio_length=args.audio_length, target_sr=args.target_sr,
            crop_mode='center',
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

        all_pesq, all_stoi, all_sisdr = [], [], []
        all_pesq_noisy, all_stoi_noisy, all_sisdr_noisy = [], [], []  # 输入对比

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                noisy_spec = batch['noisy_spec'].to(device)
                noisy_wav = batch['noisy_wav'].numpy()[0]
                clean_wav = batch['clean_wav'].numpy()[0]
                sample_name = batch['sample_name'][0]

                # 能量归一化系数（用于还原）
                c = np.sqrt(noisy_wav.size / (np.sum(noisy_wav ** 2) + 1e-8))

                # 模型推理
                noisy_input = power_compress(noisy_spec.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)

                with torch.cuda.amp.autocast():
                    est_real, est_imag, _ = model(noisy_input)

                # 解压 + ISTFT 重建增强波形
                est_real_f = est_real.permute(0, 1, 3, 2)  # (B, 1, F, T)
                est_imag_f = est_imag.permute(0, 1, 3, 2)
                est_spec_uncompress = power_uncompress(est_real_f, est_imag_f).squeeze(1)
                est_spec_complex = torch.complex(est_spec_uncompress[..., 0], est_spec_uncompress[..., 1])
                est_audio = torch.istft(
                    est_spec_complex, n_fft, hop,
                    window=window.to(device), onesided=True,
                )
                est_wav = est_audio.cpu().numpy()[0]

                # 对齐长度
                min_len = min(len(est_wav), len(clean_wav), len(noisy_wav))
                est_wav = est_wav[:min_len]
                clean_wav = clean_wav[:min_len]
                noisy_wav = noisy_wav[:min_len]

                # 还原能量归一化
                est_wav = est_wav / (c + 1e-8)
                clean_wav = clean_wav / (c + 1e-8)
                noisy_wav = noisy_wav / (c + 1e-8)

                # 计算指标
                try:
                    p = calc_pesq(args.target_sr, clean_wav, est_wav, 'wb')
                except Exception:
                    p = float('nan')
                s = calc_stoi(clean_wav, est_wav, args.target_sr)
                d = si_sdr(est_wav, clean_wav)

                # 输入的指标（noisy vs clean）
                try:
                    p_noisy = calc_pesq(args.target_sr, clean_wav, noisy_wav, 'wb')
                except Exception:
                    p_noisy = float('nan')
                s_noisy = calc_stoi(clean_wav, noisy_wav, args.target_sr)
                d_noisy = si_sdr(noisy_wav, clean_wav)

                all_pesq.append(p)
                all_stoi.append(s)
                all_sisdr.append(d)
                all_pesq_noisy.append(p_noisy)
                all_stoi_noisy.append(s_noisy)
                all_sisdr_noisy.append(d_noisy)

                # 保存部分音频
                if args.save_wav and idx < args.max_wav_save:
                    torchaudio.save(
                        os.path.join(wav_dir, f'{test_split}_{sample_name}_enhanced.wav'),
                        torch.FloatTensor(est_wav).unsqueeze(0), args.target_sr,
                    )
                    torchaudio.save(
                        os.path.join(wav_dir, f'{test_split}_{sample_name}_noisy.wav'),
                        torch.FloatTensor(noisy_wav).unsqueeze(0), args.target_sr,
                    )
                    torchaudio.save(
                        os.path.join(wav_dir, f'{test_split}_{sample_name}_clean.wav'),
                        torch.FloatTensor(clean_wav).unsqueeze(0), args.target_sr,
                    )

                if (idx + 1) % 100 == 0:
                    print(f'  {idx + 1}/{len(ds)} 样本已处理')

        # 汇总
        n_valid = sum(1 for x in all_pesq if not np.isnan(x))
        avg_pesq = np.nanmean(all_pesq)
        avg_stoi = np.nanmean(all_stoi)
        avg_sisdr = np.nanmean(all_sisdr)
        avg_pesq_noisy = np.nanmean(all_pesq_noisy)
        avg_stoi_noisy = np.nanmean(all_stoi_noisy)
        avg_sisdr_noisy = np.nanmean(all_sisdr_noisy)

        result = {
            'test_split': test_split,
            'num_samples': len(ds),
            'enhanced': {
                'PESQ': round(avg_pesq, 4),
                'STOI': round(avg_stoi, 4),
                'SI_SDR_dB': round(avg_sisdr, 2),
            },
            'noisy_input': {
                'PESQ': round(avg_pesq_noisy, 4),
                'STOI': round(avg_stoi_noisy, 4),
                'SI_SDR_dB': round(avg_sisdr_noisy, 2),
            },
            'improvement': {
                'PESQ': round(avg_pesq - avg_pesq_noisy, 4),
                'STOI': round(avg_stoi - avg_stoi_noisy, 4),
                'SI_SDR_dB': round(avg_sisdr - avg_sisdr_noisy, 2),
            },
        }
        summary[test_split] = result

        print(f'  增强:  PESQ={avg_pesq:.3f} | STOI={avg_stoi:.3f} | SI-SDR={avg_sisdr:.2f} dB')
        print(f'  输入:  PESQ={avg_pesq_noisy:.3f} | STOI={avg_stoi_noisy:.3f} | SI-SDR={avg_sisdr_noisy:.2f} dB')
        print(f'  提升:  PESQ={avg_pesq - avg_pesq_noisy:+.3f} | STOI={avg_stoi - avg_stoi_noisy:+.3f} | SI-SDR={avg_sisdr - avg_sisdr_noisy:+.2f} dB')

    # 总体平均
    print(f'\n{"=" * 60}')
    print('四测试集平均')
    print(f'{"=" * 60}')
    avg = {
        'enhanced': {
            'PESQ': round(np.mean([summary[t]['enhanced']['PESQ'] for t in ['test1','test2','test3','test4']]), 4),
            'STOI': round(np.mean([summary[t]['enhanced']['STOI'] for t in ['test1','test2','test3','test4']]), 4),
            'SI_SDR_dB': round(np.mean([summary[t]['enhanced']['SI_SDR_dB'] for t in ['test1','test2','test3','test4']]), 2),
        },
        'noisy_input': {
            'PESQ': round(np.mean([summary[t]['noisy_input']['PESQ'] for t in ['test1','test2','test3','test4']]), 4),
            'STOI': round(np.mean([summary[t]['noisy_input']['STOI'] for t in ['test1','test2','test3','test4']]), 4),
            'SI_SDR_dB': round(np.mean([summary[t]['noisy_input']['SI_SDR_dB'] for t in ['test1','test2','test3','test4']]), 2),
        },
        'improvement': {
            'PESQ': round(np.mean([summary[t]['improvement']['PESQ'] for t in ['test1','test2','test3','test4']]), 4),
            'STOI': round(np.mean([summary[t]['improvement']['STOI'] for t in ['test1','test2','test3','test4']]), 4),
            'SI_SDR_dB': round(np.mean([summary[t]['improvement']['SI_SDR_dB'] for t in ['test1','test2','test3','test4']]), 2),
        },
    }
    print(f'  增强:  PESQ={avg["enhanced"]["PESQ"]:.3f} | STOI={avg["enhanced"]["STOI"]:.3f} | SI-SDR={avg["enhanced"]["SI_SDR_dB"]:.2f} dB')
    print(f'  输入:  PESQ={avg["noisy_input"]["PESQ"]:.3f} | STOI={avg["noisy_input"]["STOI"]:.3f} | SI-SDR={avg["noisy_input"]["SI_SDR_dB"]:.2f} dB')
    print(f'  提升:  PESQ={avg["improvement"]["PESQ"]:+.3f} | STOI={avg["improvement"]["STOI"]:+.3f} | SI-SDR={avg["improvement"]["SI_SDR_dB"]:+.2f} dB')

    # 保存
    out_path = os.path.join(args.save_dir, 'denoise_evaluation.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'average': avg}, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='CMGAN 去噪评估')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_root', type=str,
                        default='/mnt/st16t/xxn/program/dataset/T60_Dataset_v7')
    parser.add_argument('--n_fft', type=int, default=400)
    parser.add_argument('--hop_length', type=int, default=100)
    parser.add_argument('--audio_length', type=float, default=4.0)
    parser.add_argument('--target_sr', type=int, default=16000)
    parser.add_argument('--n_tscb', type=int, default=4, choices=[2, 4])
    parser.add_argument('--save_dir', type=str, default='test_results')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_wav', action='store_true', help='保存增强后的音频文件')
    parser.add_argument('--max_wav_save', type=int, default=5, help='每个测试集最多保存的音频数')
    return parser.parse_args()


if __name__ == '__main__':
    run_denoise_test(parse_args())
