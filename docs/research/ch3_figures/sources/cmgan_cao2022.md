# CMGAN (Cao et al., Interspeech 2022)

- Paper/PDF: [CMGAN: Conformer-based Metric GAN for Speech Enhancement](https://www.isca-archive.org/interspeech_2022/cao22_interspeech.pdf)
- **Figure 1(a), “Encoder-decoders generator architecture”** is the closest template for the proposed concept figure. It shows STFT/power-compressed magnitude and complex components entering an encoder, a TS-Conformer stack, two parallel decoder paths, complex recombination, and iSTFT. The figure deliberately keeps the top-level path compact while labeling only the principal intermediate tensors.
- **Figure 1(b), “Two-stage conformer (TS-Conformer)”** shows the reshape–time-Conformer–residual–reshape–frequency-Conformer sequence. This is a strong template for the requested TS-Conformer/TSCB detail figure: separate time and frequency stages, explicit residual additions, and no full tensor dimensions in every internal sub-block.
- **Figure 1(c)** shows the metric discriminator, which is not needed for the current multitask model but demonstrates CMGAN’s visual vocabulary.
- The paper’s decoder description (Sec. 2.1.3) says both Mask Decoder and Complex Decoder use a dilated DenseNet plus sub-pixel convolution to restore frequency resolution. The mask path ends in a channel-squeeze convolution and PReLU mask output; the complex path has the same topology but **no activation** on the unbounded complex output.
- The paper’s encoder description (Sec. 2.1.1) specifies convolution + instance normalization + PReLU blocks around a four-layer dilated DenseNet (dilation {1,2,4,8}); this is a suitable basis for the requested denoise encoder detail figure.

## Directly reusable drawing choices

1. Use one compact top-level generator figure with STFT at the input and iSTFT at the output, following Fig. 1(a).
2. Draw TS-Conformer as a two-lane sequential block (time then frequency), following Fig. 1(b), with residual/add nodes visible.
3. Combine the two decoder lanes into one enclosing “Denoise Decoder” box in the concept figure, but keep Mask Decoder and Complex Decoder as labeled sub-branches in the detail figure.
4. Represent PReLU as a small rounded rectangle labeled `PReLU` (the CMGAN legend uses text labels rather than a triangular activation symbol). Keep the final complex-output block activation-free.

