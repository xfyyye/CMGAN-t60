# CGA-MGAN (speech enhancement architecture)
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10137386/

The paper explicitly presents an encoder-decoder generator in Figure 6, a detailed gated block in Figure 7, a two-stage time/frequency block in Figure 8, and a CGAU detail in Figure 9. Its prose describes three decoder blocks: one magnitude-mask decoder and two complex-spectrum decoders, each ending in a channel-compressing Conv2D. Encoder blocks are Conv2D + instance normalization + PReLU.

Useful convention: one concept-level generator diagram followed by focused module details, with the mask and complex outputs shown as parallel branches and merged through complex reconstruction before iSTFT. This is close to the requested denoising branch.

Evidence excerpt (source text): “The architecture of the generator is shown in Figure 6.”
