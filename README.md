# CMGAN-T60 — Blind T60 Reverberation Time Estimation

Adapts the [CMGAN](https://github.com/ruizhecao96/CMGAN) (Conformer-Based Metric-GAN) speech-enhancement backbone into a **single-task T60 regression framework**. The enhancement decoder and discriminator are removed; only the DenseEncoder + TSCB acoustic backbone is kept, with a lightweight `T60HeadWithPool` regression head added on top.

> **Branch**: `exp/single-t60` — focused exclusively on T60 estimation from noisy reverberant speech. See [`AGENTS.md`](AGENTS.md) for full design decisions.

---

## Results (CMGAN\_t60\_2tscb\_v2, 2-TSCB model)

| Test Set | RMSE (ms) | MAE (ms) | Pearson *r* | R² |
|---|---|---|---|---|
| test1 — simulated RIR + seen noise | 109.73 | 67.99 | 0.9483 | 0.8961 |
| test2 — real RIR + seen noise | 118.82 | 75.19 | 0.9383 | 0.8783 |
| test3 — simulated RIR + unseen noise | 110.50 | 73.92 | 0.9505 | 0.8951 |
| test4 — real RIR + unseen noise | 113.33 | 73.83 | 0.9448 | 0.8877 |
| **Average** | **113.10** | **72.73** | **0.9455** | **0.8893** |

---

## Architecture

```
wav (B, 64000)
 → STFT(n_fft=400, hop=100) → (B, 2, 641, 201)  [real, imag]
 → DenseEncoder              → (B, 64, 641, 101)  [F stride=2]
 → TSCB × N                  → (B, 64, 641, 101)  [N=2 or 4]
 → T60HeadWithPool
     mean(F) → MultiStatsPool(avg+max+std, T) → FC(192→128→64→1) → Sigmoid
 → t60_pred (B,)  [normalized to 0–1]
```

| Variant | TSCB Layers | Parameters |
|---|---|---|
| `TSCNet_T60Estimator` | 4 | ~1.87 M |
| `TSCNet_T60Estimator_2TSCB` | 2 | ~1.35 M |

---

## Requirements

- Python ≥ 3.9
- CUDA 11.8 / 12.x (GPU training)

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `torchaudio>=2.0`, `einops>=0.6`, `scipy>=1.10`, `soundfile>=0.12`, `PyYAML>=6.0`, `swanlab>=0.3` (optional experiment tracking).

---

## Dataset

`T60_Dataset_v7` — 23 GB, 16 kHz mono, ~4 s clips:

```
T60_Dataset_v7/
  train/      40,000 samples
  eval/        5,136 samples
  test1/       1,080 samples  (simulated RIR + seen noise)
  test2/       1,080 samples  (real RIR + seen noise)
  test3/       1,080 samples  (simulated RIR + unseen noise)
  test4/       1,080 samples  (real RIR + unseen noise)
```

Each sample directory name encodes its label:
```
speech000001_reverb_0.214_0dB/   →  T60 = 0.214 s,  SNR = 0 dB
```

Set the path in `configs/t60_single/*.yaml` under `data.dataset_root`, or pass `-d /path/to/T60_Dataset_v7` at runtime.

---

## Quick Start

### Training

```bash
conda activate demucs_xxn

# Single config (GPU 0,1)
./train.sh -c configs/t60_single/mlp_mse.yaml -g 0,1

# Train + auto-evaluate test1-test4 immediately after
./train-and-test.sh -c configs/t60_single/mlp_mse.yaml -g 0,1

# Override dataset path without editing YAML
./train.sh -c configs/t60_single/mlp_mse.yaml -d /path/to/T60_Dataset_v7 -g 0,1

# Pass extra train.py args after --
./train.sh -c configs/t60_single/mlp_mse.yaml -- --batch_size 2 --accum_steps 16
```

Run multiple loss experiments in parallel:

```bash
./train.sh -c configs/t60_single/mlp_mse.yaml   -g 0,1 &
./train.sh -c configs/t60_single/mlp_mae.yaml   -g 2,3 &
./train.sh -c configs/t60_single/mlp_huber.yaml -g 4,5 &
```

### Evaluation only

```bash
./test.sh -c configs/t60_single/mlp_mse.yaml \
          -m runs/single_t60_mlp_mse/best_model.pth \
          -g 0
```

---

## Configuration

All hyperparameters live in `configs/t60_single/*.yaml`. Key fields:

```yaml
experiment:
  name: single_t60_mlp_mse
  output_root: runs

model:
  n_tscb: 2           # 2 (faster) or 4 (full CMGAN depth)

data:
  dataset_root: /path/to/T60_Dataset_v7
  n_fft: 400
  hop_length: 100
  audio_length: 4.0
  target_sr: 16000
  t60_min: 0.1
  t60_max: 1.5

train:
  batch_size: 4       # adjust for available VRAM
  accum_steps: 8      # effective batch = batch_size × accum_steps = 32
  max_epochs: 100
  lr: 0.0005
  weight_decay: 0.00001
  early_stop_patience: 15
  num_workers: 4

loss:
  name: mse           # mse | mae | huber

logging:
  swanlab_project: T60_Estimation
```

---

## Default Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `batch_size` | 4 | Conformer O(T²) memory limit |
| `accum_steps` | 8 | Effective batch = 32 |
| `lr` | 5e-4 | AdamW |
| `weight_decay` | 1e-5 | AdamW |
| `max_epochs` | 100 | With early stopping |
| `early_stop_patience` | 15 | Monitored on val loss |
| `n_fft` / `hop` | 400 / 100 | ~641 frames @ 4 s / 16 kHz |
| `audio_length` | 4.0 s | Fixed-length input |
| T60 range | 0.1 – 1.5 s | Min-max normalized to [0, 1] |
| Loss | MSE | `mse` / `mae` / `huber` |

---

## Output Structure

```
runs/{experiment_name}/
  best_model.pth          # checkpoint with best val loss
  training_history.json   # per-epoch train/val metrics
  test/
    evaluation_results_test{1-4}.json   # per-sample + grouped metrics
    evaluation_summary.json             # cross-test average
```

Evaluation JSONs are compatible with the `/compare` skill's **Format B**.

---

## Evaluation Metrics

For each of test1–test4:

- **Overall**: RMSE (ms), MAE (ms), Bias (ms), Pearson *r*, R², Mean Relative Error (%)
- **T60 bins**: `[0.1–0.3, 0.3–0.5, …, 1.3–1.5]` s — RMSE / MAE / *r* per bin
- **SNR groups**: `[-5, 0, 5, 10, 15, 20]` dB — RMSE / MAE / *r* per group

---

## File Overview

| File | Description |
|---|---|
| `models/conformer.py` | Conformer block (upstream CMGAN, unmodified) |
| `models/generator.py` | DenseEncoder + TSCB (upstream CMGAN, unmodified) |
| `models/generator_t60.py` | `TSCNet_T60Estimator` + `T60HeadWithPool` (new) |
| `dataset.py` | `T60_Dataset_v7` loader — complex STFT + T60 label |
| `train.py` | Single-task training loop — AMP + gradient accumulation + SwanLab |
| `test.py` | test1–test4 evaluation with grouped metrics |
| `utils.py` | `power_compress`, `kaiming_init`, `LearnableSigmoid` |
| `configs/t60_single/` | YAML configs for MSE / MAE / Huber experiments |
| `train.sh` | Training launcher |
| `train-and-test.sh` | Train then auto-evaluate |
| `test.sh` | Evaluation-only launcher |

---

## Design Notes

| Decision | Choice | Reason |
|---|---|---|
| Input | Raw complex STFT (real + imag + mag → 3ch) | Preserves phase information |
| Energy normalization | `c = sqrt(T / Σwav²)` | Consistent with upstream CMGAN |
| T60 normalization | Min-max [0.1, 1.5] → [0, 1] | Matches Sigmoid output range |
| GAN discriminator | **Removed** | Not needed for regression |
| Enhancement decoder | **Removed** | Single-task branch |
| AMP | Enabled | Conformer attention is O(T²); 4 s audio needs FP16 |
| Gradient accumulation | 8 steps (default) | Effective batch = 32 with batch_size=4 |

### Memory Note

Conformer self-attention is O(T²). With T=641 and 4-layer TSCB:

- Each TSCB has 2 attention passes (time + frequency)
- 4 layers × 2 = 8 attention operations per forward pass
- GPU memory can be tight; reduce `batch_size` or `--audio_length` if OOM

---

## Acknowledgements

Built on top of [CMGAN](https://github.com/ruizhecao96/CMGAN) by Ruizhe Cao et al.

```bibtex
@inproceedings{cao2022cmgan,
  title     = {CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement},
  author    = {Ruizhe Cao and Sherif Abdulatif and Bin Yang},
  booktitle = {Interspeech},
  year      = {2022}
}
```
