"""
CMGAN T60 多任务数据集
输入格式: 复数 STFT (B, 2, T, F)，不做功率压缩
多任务: T60 回归 + 去噪 (noisy_reverb → clean_reverb)
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

DEFAULT_DATASET_ROOT = os.environ.get('T60_DATASET_ROOT')


def resolve_dataset_root(dataset_root=None):
    root = dataset_root or os.environ.get('T60_DATASET_ROOT')
    if not root:
        raise ValueError('未指定数据集路径: 请设置 T60_DATASET_ROOT 或传入 --dataset_root/-d')
    return os.path.abspath(os.path.expanduser(root))


# ─── 标签提取 ────────────────────────────────────────────────────

def parse_sample_name(dirname: str):
    """speech000001_reverb_0.214_0dB → t60=0.214, snr=0"""
    parts = dirname.split('_')
    t60 = float(parts[-2])
    snr = float(parts[-1].replace('dB', ''))
    return t60, snr


class T60Normalizer:
    """T60 min-max 归一化 [0.1, 1.5] → [0, 1]"""

    def __init__(self, t60_min=0.1, t60_max=1.5):
        self.t60_min = t60_min
        self.t60_max = t60_max

    def normalize(self, t60):
        return (t60 - self.t60_min) / (self.t60_max - self.t60_min)

    def denormalize(self, norm_t60):
        return norm_t60 * (self.t60_max - self.t60_min) + self.t60_min

    def __call__(self, t60):
        return self.normalize(t60)


# ─── 数据集 ──────────────────────────────────────────────────────

class CMGANT60Dataset(Dataset):
    """CMGAN T60 多任务数据集

    返回:
        noisy_spec:  (2, T, F) 含噪混响语音的复数STFT [real, imag]
        clean_spec:  (2, T, F) 纯混响语音的复数STFT (去噪target)
        t60:         (,) 归一化T60值
        t60_raw:     (,) 原始T60值
        snr:         (,) SNR值
        sample_name: str
        noisy_wav:   (T_wav,) 归一化后的波形 (用于时域损失)
        clean_wav:   (T_wav,) 归一化后的波形
    """

    SPLIT_MAP = {'val': 'eval', 'dev': 'eval'}

    def __init__(
        self,
        dataset_root=DEFAULT_DATASET_ROOT,
        split='train',
        n_fft=400,
        hop_length=100,
        audio_length=4.0,
        target_sr=16000,
        t60_range=(0.1, 1.5),
    ):
        self.dataset_root = resolve_dataset_root(dataset_root)
        self.split = self.SPLIT_MAP.get(split, split)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.target_samples = int(audio_length * target_sr)
        self.t60_normalizer = T60Normalizer(*t60_range)
        self.split_dir = os.path.join(self.dataset_root, self.split)
        self.register_buffer = None  # 延迟创建
        self.samples = self._scan_samples()

    def _scan_samples(self):
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f'数据集 split 目录不存在: {self.split_dir}')

        samples = []
        for dirname in sorted(os.listdir(self.split_dir)):
            dirpath = os.path.join(self.split_dir, dirname)
            if not os.path.isdir(dirpath):
                continue
            main_wav = os.path.join(dirpath, f'{dirname}.wav')
            if os.path.exists(main_wav):
                samples.append(dirname)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, filepath):
        wav, sr = torchaudio.load(filepath)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0)

    def _ensure_audio_length(self, wav, filepath):
        if wav.shape[0] != self.target_samples:
            raise ValueError(
                f'音频长度不匹配: {filepath}, '
                f'expected {self.target_samples} samples, got {wav.shape[0]}'
            )
        return wav

    def _energy_normalize(self, wav):
        """能量归一化: c = sqrt(T / sum(wav^2))"""
        c = torch.sqrt(wav.size(-1) / (torch.sum(wav ** 2.0, dim=-1) + 1e-8))
        return wav * c

    def _wav_to_complex_stft(self, wav):
        """波形 → 复数STFT → (2, T, F) [real, imag]
        不做功率压缩，保留原始频谱
        """
        spec = torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            window=torch.hamming_window(self.n_fft),
            onesided=True,
            return_complex=True,
        )  # (F, T) complex
        real = spec.real.permute(1, 0)  # (T, F)
        imag = spec.imag.permute(1, 0)  # (T, F)
        return torch.stack([real, imag], dim=0)  # (2, T, F)

    def __getitem__(self, idx):
        dirname = self.samples[idx]
        dirpath = os.path.join(self.split_dir, dirname)
        t60, snr = parse_sample_name(dirname)

        # 加载含噪混响音频
        noisy_path = os.path.join(dirpath, f'{dirname}.wav')
        noisy_wav = self._load_audio(noisy_path)
        noisy_wav = self._ensure_audio_length(noisy_wav, noisy_path)

        # 加载纯混响音频（去噪target）
        clean_path = os.path.join(dirpath, f'{dirname}_denoised.wav')
        clean_wav = self._load_audio(clean_path) if os.path.exists(clean_path) else torch.zeros_like(noisy_wav)
        clean_wav = self._ensure_audio_length(clean_wav, clean_path)

        # 能量归一化（对 noisy 和 clean 使用相同的系数）
        c = torch.sqrt(noisy_wav.size(-1) / (torch.sum(noisy_wav ** 2.0, dim=-1) + 1e-8))
        noisy_wav = noisy_wav * c
        clean_wav = clean_wav * c

        # 转复数STFT
        noisy_spec = self._wav_to_complex_stft(noisy_wav)  # (2, T, F)
        clean_spec = self._wav_to_complex_stft(clean_wav)  # (2, T, F)

        return {
            'noisy_spec': noisy_spec,
            'clean_spec': clean_spec,
            'noisy_wav': noisy_wav,
            'clean_wav': clean_wav,
            't60': torch.tensor(self.t60_normalizer(t60), dtype=torch.float32),
            't60_raw': torch.tensor(t60, dtype=torch.float32),
            'snr': torch.tensor(snr, dtype=torch.float32),
            'sample_name': dirname,
        }


# ─── Collate ──────────────────────────────────────────────────────

def collate_fn(batch):
    """处理变长STFT：对齐到batch内最大T和F"""
    max_t = max(item['noisy_spec'].shape[1] for item in batch)
    max_f = max(item['noisy_spec'].shape[2] for item in batch)

    result = {}
    for key in ['noisy_spec', 'clean_spec']:
        tensors = []
        for item in batch:
            t = item[key]
            pad_t = max_t - t.shape[1]
            pad_f = max_f - t.shape[2]
            if pad_t > 0 or pad_f > 0:
                t = torch.nn.functional.pad(t, (0, pad_f, 0, pad_t))
            tensors.append(t)
        result[key] = torch.stack(tensors)

    for key in ['noisy_wav', 'clean_wav']:
        max_len = max(item[key].shape[0] for item in batch)
        tensors = []
        for item in batch:
            t = item[key]
            if t.shape[0] < max_len:
                t = torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
            tensors.append(t)
        result[key] = torch.stack(tensors)

    for key in ['t60', 't60_raw', 'snr']:
        result[key] = torch.stack([item[key] for item in batch])

    result['sample_name'] = [item['sample_name'] for item in batch]
    return result


# ─── DataLoader 工厂 ──────────────────────────────────────────────

def create_dataloaders(
    dataset_root=DEFAULT_DATASET_ROOT,
    n_fft=400,
    hop_length=100,
    batch_size=8,
    num_workers=4,
    audio_length=4.0,
    target_sr=16000,
    t60_range=(0.1, 1.5),
    pin_memory=True,
):
    common = dict(
        dataset_root=dataset_root,
        n_fft=n_fft,
        hop_length=hop_length,
        audio_length=audio_length,
        target_sr=target_sr,
        t60_range=t60_range,
    )

    train_ds = CMGANT60Dataset(split='train', **common)
    eval_ds = CMGANT60Dataset(split='eval', **common)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return train_loader, eval_loader


def create_test_loader(
    test_split='test1',
    dataset_root=DEFAULT_DATASET_ROOT,
    n_fft=400,
    hop_length=100,
    batch_size=8,
    num_workers=4,
    audio_length=4.0,
    target_sr=16000,
    t60_range=(0.1, 1.5),
):
    ds = CMGANT60Dataset(
        dataset_root=dataset_root, split=test_split,
        n_fft=n_fft, hop_length=hop_length,
        audio_length=audio_length, target_sr=target_sr,
        t60_range=t60_range,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
