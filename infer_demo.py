"""
KAN 多任务模型推理 demo：对选定样本生成含噪、干净、增强后三路音频，供主观试听。

输出目录: demo_audio/
  {sample_name}_noisy.wav     — 含噪混响输入（能量归一化后）
  {sample_name}_clean.wav     — 纯混响 target（*_denoised.wav，无噪声）
  {sample_name}_enhanced.wav  — 模型增强输出（peak normalized 至 0.98）

注意：clean 是 *_denoised.wav（去噪 target），仍含混响，不是干声。

Usage:
    # 使用默认配置（w_denoise checkpoint，test1 中的 4 个代表性样本）
    CUDA_VISIBLE_DEVICES=5 python infer_demo.py

    # 修改脚本顶部的 CKPT / DATASET / SAMPLES 常量可切换实验和样本

推理 pipeline 与 test_multitask_denoise.py 完全一致：
    noisy wav → 能量归一化 → STFT → power_compress
    → permute(0,3,2,1) → permute(0,1,3,2) → model
    → permute(0,1,3,2) → power_uncompress → iSTFT → 还原能量归一化 → peak normalize
"""

import os, sys, torch, torchaudio, math
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent
CKPT    = PROJECT / "runs/kan_multitask_w_denoise/best_model.pth"
OUT_DIR = PROJECT / "demo_audio"
DATASET = Path("/mnt/tidal-sh01/usr/chuan/youling/project/dataset/T60_Dataset_v7_4s/test1")
OUT_DIR.mkdir(exist_ok=True)

# ── STFT params (must match training) ─────────────────────────────────────
N_FFT   = 400
HOP     = 100
SR      = 16000
AUDIO_LEN = 4.0  # seconds

# ── representative samples ─────────────────────────────────────────────────
# (folder_name, label)
SAMPLES = [
    ("speech000005_reverb_0.162_-5dB",  "short_T60_low_SNR"),   # T60=0.16s, SNR=-5dB  (hardest)
    ("speech000006_reverb_0.551_0dB",   "mid_T60_low_SNR"),     # T60=0.55s, SNR=0dB
    ("speech000002_reverb_1.168_5dB",   "long_T60_mid_SNR"),    # T60=1.17s, SNR=5dB
    ("speech000001_reverb_0.294_20dB",  "short_T60_high_SNR"),  # T60=0.29s, SNR=20dB  (easiest)
]

# ── load model ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT))
from models.generator_t60 import TSCNet_KAN_MultiTask, TSCNet_KAN_MultiTask_2TSCB
from utils import power_compress, power_uncompress

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
# strip DataParallel prefix if present
state = {k.replace("module.", ""): v for k, v in state.items()}

cfg = ckpt.get("config", {})
model_cfg = cfg.get("model", {})
n_tscb = model_cfg.get("n_tscb", 2)
model_cls = TSCNet_KAN_MultiTask_2TSCB if n_tscb == 2 else TSCNet_KAN_MultiTask
model_kwargs = dict(
    t60_head_type=model_cfg.get("t60_head_type", "fourier_kan"),
    t60_fourier_proj_dim=model_cfg.get("t60_fourier_proj_dim", 64),
    t60_fourier_hidden_dims=model_cfg.get("t60_fourier_hidden_dims", [32, 16]),
    t60_fourier_first_num_frequencies=model_cfg.get("t60_fourier_first_num_frequencies", 16),
    t60_fourier_hidden_num_frequencies=model_cfg.get("t60_fourier_hidden_num_frequencies", 8),
    t60_fourier_dropout=model_cfg.get("t60_fourier_dropout", 0.1),
    t60_out_activation=model_cfg.get("t60_out_activation", "sigmoid"),
)
model = model_cls(**model_kwargs)
model.load_state_dict(state)
model.eval().to(device)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# ── helpers ────────────────────────────────────────────────────────────────
def load_wav(path, target_len):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0)  # mono
    n = int(target_len * SR)
    if wav.shape[0] < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
    else:
        wav = wav[:n]
    return wav  # (T,)

def enhance(noisy_wav):
    """noisy_wav: (T,) tensor on cpu → enhanced (T,) on cpu"""
    wav = noisy_wav.to(device)
    # energy norm (same as training)
    c = math.sqrt(wav.shape[0] / (wav ** 2).sum().clamp(min=1e-8))
    wav = wav * c

    # STFT → (F=201, T=641)
    window = torch.hann_window(N_FFT).to(device)
    spec = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)

    # Replicate test_multitask_denoise.py pipeline exactly.
    # dataset noisy_spec shape: (2, T, F).  After batching: (B, 2, T, F).
    # → permute(0,3,2,1) → (B, F, T, 2)
    # → power_compress   → (B, 2, F, T)
    # → permute(0,1,3,2) → (B, 2, T, F)   ← model input
    noisy_spec = torch.stack([spec.real, spec.imag], dim=0)   # (2, F, T)
    noisy_spec = noisy_spec.permute(1, 2, 0).unsqueeze(0)     # (1, F, T, 2)  ← (B, F, T, 2) equivalent
    noisy_input = power_compress(noisy_spec).permute(0, 1, 3, 2)  # (1, 2, T, F)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            est_real, est_imag, _ = model(noisy_input)   # (1, 1, T, F) each

    # Decode: permute T/F back, uncompress, istft
    est_real_f = est_real.permute(0, 1, 3, 2)            # (1, 1, F, T)
    est_imag_f = est_imag.permute(0, 1, 3, 2)
    est_uncomp = power_uncompress(est_real_f, est_imag_f).squeeze(1)  # (1, F, T, 2)
    est_spec = torch.complex(est_uncomp[..., 0], est_uncomp[..., 1])  # (1, F, T)
    enhanced_wav = torch.istft(est_spec, N_FFT, HOP, window=window,
                               length=noisy_wav.shape[0])              # (1, samples)
    # undo energy norm
    enhanced_wav = enhanced_wav / (c + 1e-8)
    return enhanced_wav.squeeze(0).cpu()

# ── run ────────────────────────────────────────────────────────────────────
print(f"\nOutput dir: {OUT_DIR}\n")
for folder, label in SAMPLES:
    sample_dir = DATASET / folder
    noisy_path = sample_dir / f"{folder}.wav"
    if not noisy_path.exists():
        print(f"[SKIP] {noisy_path} not found")
        continue

    noisy = load_wav(noisy_path, AUDIO_LEN)
    enhanced = enhance(noisy)

    # save
    torchaudio.save(str(OUT_DIR / f"{label}_noisy.wav"),   noisy.unsqueeze(0),    SR)
    torchaudio.save(str(OUT_DIR / f"{label}_enhanced.wav"), enhanced.unsqueeze(0), SR)

    # quick stats
    pesq_note = ""
    try:
        from pesq import pesq as pesq_fn
        from pesq import PesqError
        p_noisy    = pesq_fn(SR, noisy.numpy(),    noisy.numpy(),    'wb')
        p_enhanced = pesq_fn(SR, noisy.numpy(), enhanced.numpy(), 'wb')
        pesq_note = f"  PESQ: {p_noisy:.2f}→{p_enhanced:.2f}"
    except Exception:
        pass

    noisy_rms    = (noisy ** 2).mean().sqrt().item()
    enhanced_rms = (enhanced ** 2).mean().sqrt().item()
    print(f"[{label}]{pesq_note}  RMS: {noisy_rms:.4f}→{enhanced_rms:.4f}")

print("\nDone. Files written to demo_audio/")
