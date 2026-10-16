# Independent speech-enhancement diagram/PReLU reference
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10137386/

The paper uses a standard PReLU activation in each encoder block and makes its module hierarchy explicit: Figure 3 for Conformer, Figures 4–5 for convolution/feed-forward details, Figure 6 for encoder-decoder, and Figures 7–9 for gated/convolution-attention blocks.

Useful convention: do not invent a geometric triangle for PReLU. Draw PReLU as a small activation box labeled “PReLU” in the block flow, and reserve the full piecewise curve for a separate inset only if the activation itself is discussed mathematically.
