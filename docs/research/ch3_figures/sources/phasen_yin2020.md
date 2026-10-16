# PHASEN (Yin et al., AAAI 2020)

- Paper/PDF: [PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network](https://ojs.aaai.org/index.php/AAAI/article/download/6489/6345)
- **Figure 2, “The proposed two-stream PHASEN architecture”** is a useful reference for visually separating magnitude/mask and phase/complex branches. It uses clearly named streams, parallel paths, and a shared output/reconstruction stage rather than hiding branch semantics inside one generic decoder box.
- **Figure 3, “Flowchart of the proposed FTBs”** (described in the paper) is a reference for making a dedicated detail figure for a nonstandard frequency-processing block: it expands the block once and uses repeated TSB/FTB markers in the overview.
- PHASEN labels branch outputs by their role (amplitude mask vs. complex phase) and makes the interaction points explicit. This supports showing Mask Decoder and Complex Decoder as two labeled sub-branches inside a single Denoise Decoder detail figure.

## Reusable choices

- Use branch-specific colors and labels (`Mask Decoder`, `Complex Decoder`) while keeping a shared neutral color for the encoder/TSCB trunk.
- Show the final recombination before iSTFT as a single explicit node; avoid drawing separate waveform paths for magnitude and complex estimates.

