# Prompt for Generating Architecture Figure (GPT Image-2)

> **Model Configuration**: `single_t60_kan_v2_mse` (TSCNet_T60Estimator_2TSCB + Fourier-KAN v2 head + MSE loss)

---

## Style Requirements

- **Layout**: Horizontal flowchart, left-to-right signal flow
- **Three color-coded regions** with light background shading:
  - **Left region (Signal Preprocessing)** — light blue
  - **Middle region (Acoustic Encoder)** — light green
  - **Right region (T60 Regression Head)** — light orange
- **Lines**: Solid arrows for data flow, dashed arrows for residual connections
- **Font**: English (academic paper language). Module names in **bold**, tensor dimensions in monospace
- **Overall style**: Consistent with IEEE/Interspeech speech processing papers — clean, minimal, no excessive decoration

---

## Complete Module Descriptions (Left → Right)

### Module 1: Input Signal

- Waveform icon with label:
  - **Noisy Reverberant Speech** x(t)
  - Sampling rate: 16 kHz, duration: 4.0 s
  - Tensor: **(B, 64000)**

### Module 2: Energy Normalization

- Small box labeled:
  - **Energy Normalization**
  - Formula: c = √(T / Σx²(t)), x̃(t) = c · x(t)
  - Output: **(B, 64000)** (unchanged)
- Purpose: Normalize inputs across different energy levels (inherited from original CMGAN design)

### Module 3: Short-Time Fourier Transform

- Box labeled:
  - **STFT**
  - Parameters: N_fft = 400, hop length = 100, window = Hamming
- Output: Complex spectrogram **(B, 2, 641, 201)**
  - 2 channels = [real, imag]
  - 641 = number of time frames T
  - 201 = number of frequency bins F = N_fft/2 + 1

### Module 4: Power Compression

- Box labeled:
  - **Power Compression**
  - Formula: |S|^0.3 (compress magnitude by exponent 0.3, preserve original phase)
- Output: **(B, 2, 641, 201)** (unchanged)
- Purpose: Compress dynamic range for better Conv2d learning

### Module 5: 3-Channel Concatenation

- Box labeled:
  - **Channel Concatenation**
  - Extract: magnitude = √(real² + imag²)
  - Concatenate [magnitude, real, imag] → 3 channels
- Output: **(B, 3, 641, 201)**

### Module 6: Dense Encoder

- **Large box**, title: **Dense Encoder**
- Internal structure (top to bottom):
  1. **Conv2d** (3→64, kernel 1×1) + InstanceNorm + PReLU
  2. **Dilated DenseNet** (depth=4)
     - Show 4 layers, each labeled with dilation rate: d = 1, 2, 4, 8
     - Per-layer: Padding → Conv2d (dense-connected channels) → InstanceNorm → PReLU
     - Dense connections between layers: output of each layer concatenated with all previous outputs along channel dimension
  3. **Conv2d** (64→64, kernel 1×3, stride (1,2)) + InstanceNorm + PReLU
     - Label: **Frequency-axis downsampling ×2**
- Input: **(B, 3, 641, 201)**
- Output: **(B, 64, 641, 101)** — F dimension halved

### Module 7: TSCB (Time-Frequency Conformer Block) × 2

- **Large box**, title: **TSCB (Temporal-Spectral Conformer Block) × 2**
- Use a single box with a "×2" repetition marker
- Each TSCB has a **dual-branch structure**, drawn as two sequential paths:

  **Upper path — Temporal Conformer**:
  - Reshape: (B, 64, T, F) → (B×F, T, 64)
  - Label: "Model temporal dependencies independently per frequency band"
  - **Conformer Block** (expanded internal structure):
    - FFN (×0.5 scale) + Residual
    - PreNorm → Multi-Head Self-Attention (4 heads, Shaw's relative positional encoding)
    - Conv Module (kernel=31, depthwise conv, GLU activation) + Residual
    - FFN (×0.5 scale) + Residual
    - LayerNorm
  - Reshape back: (B, 64, T, F)

  **Lower path — Spectral Conformer**:
  - Reshape: (B, 64, T, F) → (B×T, F, 64)
  - Label: "Model spectral patterns independently per time frame"
  - **Conformer Block** (same structure as above)
  - Reshape back: (B, 64, T, F)

  - **Residual connections** between branches:
    - x_t = TimeConformer(x) + x
    - x_f = FreqConformer(x_t) + x_t

- Input: **(B, 64, 641, 101)**
- Output: **(B, 64, 641, 101)** (unchanged)

### Module 8: Frequency-Average Pooling

- Small box labeled:
  - **Mean Pooling (Frequency Axis)**
  - x̄ = mean(x, dim=F)
- Input: **(B, 64, 641, 101)**
- Output: **(B, 64, 641)** — frequency dimension removed

### Module 9: Multi-Statistics Temporal Pooling

- Box labeled:
  - **Multi-Statistics Pooling (Time Axis)**
  - Compute three statistics along time dimension T and concatenate:
    - μ = mean(x, dim=T) — mean
    - max = max(x, dim=T) — maximum
    - σ = std(x, dim=T) — standard deviation
  - Concatenate [μ ⊕ max ⊕ σ]
- Input: **(B, 64, 641)**
- Output: **(B, 192)** — 64 × 3 = 192

### Module 10: Linear Projection

- Box labeled:
  - **Linear Projection**
  - Linear(192 → 64) + LayerNorm
  - (Note: v2 version has **no** Tanh activation)
- Input: **(B, 192)**
- Output: **(B, 64)**

### Module 11: Fourier-KAN Block — Core Contribution

- **Highlight with prominent border/color** (this is the paper's novel contribution)
- Title: **Fourier-KAN Regression Block**

- Internal structure — two FourierKANLayers:

  **Layer 1**:
  - Label: **FourierKAN Layer 1** (Ω = 16)
  - Formula (display beside the layer):
    - y_j = Σᵢ Σω [a_{j,i,ω} · cos(ω · xᵢ) + b_{j,i,ω} · sin(ω · xᵢ)] + biasⱼ
    - where ω = 1, 2, ..., 16
  - Followed by: LayerNorm + Dropout(0.1)
  - Input: **(B, 64)**, Output: **(B, 32)**

  **Layer 2**:
  - Label: **FourierKAN Layer 2** (Ω = 8)
  - Same formula, ω = 1, 2, ..., 8
  - Followed by: LayerNorm + Dropout(0.1)
  - Input: **(B, 32)**, Output: **(B, 16)**

- **Add annotation box beside this module**:
  - *"Hierarchical frequency design: the first layer employs a larger Ω=16 to capture fine-grained nonlinear patterns in the acoustic embedding; subsequent layers adopt a smaller Ω=8 to control parameter count, following the layered gridsize design of Fourier-ASR."*

### Module 12: Output Layer

- Box labeled:
  - **Output Layer**
  - Linear(16 → 1) + Sigmoid
- Output: **(B,)** — normalized T60 prediction ∈ [0, 1]

### Module 13: Denormalization

- Small box labeled:
  - **Min-Max Denormalization**
  - T̂₆₀ = ŷ × (T₆₀_max − T₆₀_min) + T₆₀_min
  - T₆₀_min = 0.1s, T₆₀_max = 1.5s
- Output: **T̂₆₀ (seconds)**, range [0.1, 1.5]

### Module 14: Training Objective (optional, place at bottom of figure)

- Dashed box at bottom:
  - **Training Objective**
  - Loss = MSE(T̂₆₀, T₆₀) = (1/N) Σ(T̂₆₀ᵢ − T₆₀ᵢ)²
  - Optimizer: AdamW (lr = 5×10⁻⁴, weight_decay = 10⁻⁵)
  - LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
  - Early Stopping: patience = 15 epochs
  - Mixed Precision Training (AMP)

---

## Parameter Summary Table (place at bottom of figure as annotation)

| Parameter | Value |
|-----------|-------|
| Audio length | 4.0 s @ 16 kHz |
| STFT | N_fft=400, hop=100, Hamming window |
| DenseEncoder channels | 64 |
| TSCB layers | 2 |
| Conformer attention heads | 4 |
| Conv Module kernel size | 31 |
| Fourier-KAN Ω | 16 (first layer) / 8 (hidden layers) |
| Fourier-KAN hidden dims | [32, 16] |
| Total model parameters | ~1.35M |
| T60 Head parameters | ~86K |

---

## Key Design Notes for the Artist

1. **Fourier-KAN Block (Module 11) is the core novelty** — use a distinct border color (e.g., orange or red outline) to visually distinguish it from the inherited CMGAN backbone modules.

2. **TSCB dual-branch structure (Module 7)** deserves expanded internal detail — show the temporal/spectral split-and-merge pattern clearly, as this is the key inherited design from CMGAN.

3. **Data flow dimensions should be annotated at every transition** where the tensor shape changes (STFT output, DenseEncoder output, pooling output, final prediction).

4. **Keep the overall figure clean** — avoid over-decorating. The style should match top-tier speech/audio conference papers (Interspeech, ICASSP).
