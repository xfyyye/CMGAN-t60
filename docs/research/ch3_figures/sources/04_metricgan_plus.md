# MetricGAN+ (speech enhancement lineage)
URL: https://www.isca-archive.org/interspeech_2021/fu21_interspeech.pdf

This is a lineage reference for CMGAN, not a template for the new multitask graph. It supports keeping STFT/ISTFT at the outside of the enhancement graph and treating the enhancement network as an internal spectral mapping. The architecture is comparatively compact, which supports a clean concept graph with only signal-domain transforms exposed.
