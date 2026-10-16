# DPDF-AEC (2024) — component-detail style reference

- Paper/PDF: [DPDF-AEC, Journal of Information Processing](https://www.jstage.jst.go.jp/article/transinf/advpub/0/advpub_2024EDL8084/_pdf)
- **Figure 1** gives a full architecture overview with a DenseEncoder, temporal/frequency processing, Mask/DF decoder, and iSTFT. It uses dashed boxes to indicate repeated blocks and keeps the overview readable by omitting low-level tensor dimensions.
- **Figure 2** is particularly useful for the requested detail figures. It contains four component panels: (a) Mask/DF Decoder, (b) DenseEncoder, (c) DeepFilter, and (d) DilatedDenseNet. The encoder panel explicitly uses `PWConv2d → LN+PReLU → DilatedDenseNet → Conv2d → LN+PReLU`; the DenseNet panel expands repeated dilated-convolution layers with dense concatenation and an `×M` repeat marker.
- The paper’s figure language uses green/blue component boxes, rounded operation blocks, and a larger enclosing dashed box for a repeated module. This is a good practical template when drawing the requested denoise encoder, mask decoder, and complex decoder detail panels.

## Reusable choices

- Adopt a single enlarged module box plus a small “repeat ×M” marker instead of drawing every repeated DenseNet layer.
- Put operation name and activation together (`Conv2d`, `LN + PReLU`) in one block, which avoids a visually noisy chain.
- Keep full dimensions out of the detail drawings unless a dimension change is central (e.g., frequency upsampling in a decoder).

