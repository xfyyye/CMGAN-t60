import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RIRDataset(Dataset):
    """CSV manifest backed RIR dataset.

    Required columns: path, t60. Optional columns: split, room_id.
    """

    def __init__(self, manifest, sample_rate=16000, rir_length=32768,
                 split="train", t60_min=0.1, t60_max=1.5):
        self.sample_rate = sample_rate
        self.rir_length = rir_length
        self.t60_min = t60_min
        self.t60_max = t60_max
        self.rows = []
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split", split) != split:
                    continue
                t60 = float(row["t60"])
                if t60_min <= t60 <= t60_max:
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No RIR rows for split={split} in {manifest}")

    def __len__(self):
        return len(self.rows)

    def _load(self, path):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("Install soundfile before loading WAV RIR data") from exc
        wav, sr = sf.read(path, always_2d=False)
        if sr != self.sample_rate:
            raise ValueError(f"sample rate mismatch for {path}: {sr} != {self.sample_rate}")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav[: self.rir_length]
        if len(wav) < self.rir_length:
            wav = np.pad(wav, (0, self.rir_length - len(wav)))
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak
        return wav

    def __getitem__(self, index):
        row = self.rows[index]
        wav = torch.from_numpy(self._load(row["path"])).float().unsqueeze(0)
        t60 = (float(row["t60"]) - self.t60_min) / (self.t60_max - self.t60_min)
        return {"rir": wav, "t60": torch.tensor([t60], dtype=torch.float32),
                "path": row["path"], "room_id": row.get("room_id", "")}

