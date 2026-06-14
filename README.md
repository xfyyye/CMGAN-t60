# CMGAN-T60 Multi-task — Blind T60 Estimation with Denoising Auxiliary Task

Adapts the [CMGAN](https://github.com/ruizhecao96/CMGAN) (Conformer-Based Metric-GAN) backbone into a **multi-task framework** for joint T60 reverberation time estimation and speech denoising. The GAN discriminator is removed; the DenseEncoder + TSCB backbone is shared between a FourierKAN T60 regression head and the original mask/complex decoders for denoising.

> **Branch**: `exp/kan-multitask` — multi-task joint training is the main line; single-task T60 experiments (`configs/t60_single/`) serve as ablation baselines.

---

## Key Results

### T60 Estimation (average over test1–test4)

| Model | RMSE (ms) ↓ | MAE (ms) ↓ | Pearson *r* ↑ | R² ↑ |
|---|---|---|---|---|
| Single-task (ablation baseline) | 107.70 | 62.14 | 0.9488 | 0.8996 |
| Multi-task w_t60 (α=0.1, β=1.0) | 107.84 | 62.79 | 0.9489 | 0.8993 |
| Multi-task w_eq (α=1.0, β=1.0) | 105.64 | 62.04 | 0.9520 | 0.9032 |
| Multi-task w_balanced (α=0.5, β=0.5) | 100.28 | 62.12 | 0.9569 | 0.9128 |
| **Multi-task w_denoise (α=1.0, β=0.1)** | **96.41** | **58.69** | **0.9594** | **0.9195** |

w_denoise achieves **−10.5% RMSE** over the single-task baseline with only +6.6% parameter overhead.

### Denoising Quality — w_denoise (average over test1–test4)

| Metric | Noisy Input | Enhanced | Δ |
|---|---|---|---|
| PESQ (wb) ↑ | 2.721 | **3.583** | +0.862 |
| STOI ↑ | 0.847 | **0.937** | +0.090 |
| SI-SDR (dB) ↑ | 15.91 | **21.83** | +5.92 |

---

## Architecture

```
wav (B, 64000)
 → STFT(n_fft=400, hop=100)  → (B, 2, T=641, F=201)
 → power_compress             → (B, 2, T, F)
 → [mag, real, imag] concat   → (B, 3, T, F)
 → DenseEncoder               → (B, 64, T, F/2=101)  ┐
 → TSCB_1, TSCB_2             → (B, 64, T, F/2)      ┤ shared backbone
      ├─► T60Head (FourierKAN) → t60_pred (B,)         ┘ main task
      ├─► MaskDecoder          → mask (B, 1, T, F)     ┐
      └─► ComplexDecoder       → complex (B, 2, T, F)  ┘ auxiliary denoising
                               → iSTFT → enhanced wav
```

| Component | Parameters |
|---|---|
| DenseEncoder | ~260K (shared) |
| TSCB × 2 | ~516K (shared) |
| T60Head (FourierKAN) | ~86K |
| MaskDecoder + ComplexDecoder | ~27K |
| **Total** | **~1,406K** |

---

## Gradient Probe Analysis

To understand *why* the denoising task improves T60 estimation, we instrument the shared backbone with gradient probes (recorded every 20 steps):

| Experiment | cos mean | neg% | r = ‖g_T60‖/‖g_denoise‖ |
|---|---|---|---|
| w_denoise | +0.030 | ~37% | **0.33** |
| w_balanced | +0.028 | ~41% | 0.89 |
| w_eq | +0.031 | ~40% | 0.93 |
| w_t60 | +0.024 | ~42% | 1.83 |

**Finding**: the gradient direction (cos ≈ 0, orthogonal, no conflict) is nearly identical across all runs. What differs is the gradient *magnitude* ratio r. T60 RMSE and r are highly correlated (Pearson r = 0.885): the lower r, the better T60 estimation. When denoising dominates the backbone (r=0.33), its rich supervision drives the encoder to learn fine-grained time-frequency representations that also benefit T60 regression.

See `EXPERIMENT_REPORT.md` and `ANALYSIS_GUIDE.md` for full analysis including linear probing, t-SNE, and CKA.

---

## Requirements

- Python ≥ 3.9, CUDA 11.8 / 12.x

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `torchaudio>=2.0`, `einops>=0.6`, `scipy>=1.10`, `soundfile>=0.12`, `PyYAML>=6.0`, `swanlab>=0.3` (optional).

---

## Dataset

`T60_Dataset_v7_4s` — 23 GB, 16 kHz mono, ~4 s clips:

```
T60_Dataset_v7_4s/
  train/      40,000 samples
  eval/        5,136 samples
  test1/       1,080 samples  (simulated RIR + seen noise)
  test2/       1,080 samples  (real RIR + seen noise)
  test3/       1,080 samples  (simulated RIR + unseen noise)
  test4/       1,080 samples  (real RIR + unseen noise)
```

Each sample directory (`speech{id}_reverb_{T60:.3f}_{SNR}dB/`) contains:
- `{name}.wav` — noisy reverberant speech (model input)
- `{name}_denoised.wav` — clean reverberant speech (denoising target)
- `{name}_noise.wav` — noise component (unused)

Set `data.dataset_root` in `configs/t60_multitask/*.yaml`.

---

## Quick Start

### Multi-task Training

```bash
PYTHON=/path/to/venv/bin/python

# Launch in background (detached from terminal)
setsid nohup bash scripts/multitask_kan_w_denoise.sh \
  > runs/kan_multitask_w_denoise/nohup.log 2>&1 &
```

### Multi-task Evaluation

```bash
# T60 estimation (test1–4)
bash scripts/test_multitask_kan_w_denoise.sh -g 0

# Denoising quality (PESQ / STOI / SI-SDR)
bash scripts/test_multitask_denoise_kan_w_denoise.sh -g 0
```

### Ablation — Single-task Baseline

```bash
bash train.sh -c configs/t60_single/kan_v2_mae_clip1_log.yaml -g 0,1
bash test.sh  -c configs/t60_single/kan_v2_mae_clip1_log.yaml \
              -m runs/single_t60_kan_v2_mae_clip1_log/best_model.pth -g 0
```

### Analysis Scripts

```bash
# Gradient probe visualisation (no GPU needed)
python analyze_gradients.py --out_dir figures/gradient

# Representation analysis (GPU required)
python analyze_representations.py --gpu 0 --n_samples 500 \
  --out_dir figures/representation
```

---

## Configuration

Multi-task configs live in `configs/t60_multitask/*.yaml`. Key fields:

```yaml
experiment:
  name: kan_multitask_w_denoise

model:
  n_tscb: 2
  t60_head_type: fourier_kan

data:
  dataset_root: /path/to/T60_Dataset_v7_4s

train:
  batch_size: 16
  accum_steps: 1
  max_epochs: 100
  lr: 0.0005
  early_stop_patience: 15

loss:
  alpha: 1.0   # denoise weight
  beta:  0.1   # T60 weight
```

---

## Output Structure

```
runs/kan_multitask_{name}/
  best_model.pth
  training_history.json
  nohup_train.log
  test/                         ← T60 evaluation results
    evaluation_results_test{1-4}.json
    evaluation_summary.json
  test_denoise/                 ← Denoising evaluation results
    denoise_evaluation.json

figures/
  gradient/                     ← Gradient probe plots (5 figures)
  representation/               ← Linear probe, t-SNE, CKA (4 figures)

demo_audio/                     ← Sample noisy / clean / enhanced audio
```

---

## File Overview

### Multi-task (main)

| File | Description |
|---|---|
| `train_multitask.py` | Multi-task training loop with gradient probes and SwanLab logging |
| `test_multitask.py` | T60 evaluation — test1–4, standard metrics + T60/SNR bins |
| `test_multitask_denoise.py` | Denoising evaluation — PESQ / STOI / SI-SDR |
| `infer_demo.py` | Inference demo — generates noisy / clean / enhanced audio |
| `analyze_gradients.py` | Gradient probe visualisation (cos/r time-series, histograms, bar charts) |
| `analyze_representations.py` | Representation analysis (linear probe, t-SNE, CKA) |
| `models/generator_t60.py` | `TSCNet_KAN_MultiTask_2TSCB` (multi-task) + `TSCNet_T60Estimator_2TSCB` (ablation) |
| `dataset.py` | `CMGANT60MultiTaskDataset` (multi-task) / `CMGANT60Dataset` (ablation) |
| `configs/t60_multitask/` | Four multi-task experiment configs |
| `scripts/` | Per-experiment train / test launch scripts |
| `EXPERIMENT_REPORT.md` | Full experiment report with all metrics and gradient analysis |
| `ANALYSIS_GUIDE.md` | How to read every analysis figure |

### Ablation (single-task)

| File | Description |
|---|---|
| `train.py` / `test.py` | Single-task training / evaluation |
| `train.sh` / `test.sh` / `train-and-test.sh` | Single-task launchers |
| `configs/t60_single/` | Single-task experiment configs |
| `runs/single_t60_*/` | Single-task checkpoints and results |

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
