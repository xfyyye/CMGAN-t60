# Fourier Neural Operator — transform/weights/inverse-transform layer schematic

- **Paper:** S. P. Kovachki et al. follow-up exposition, *Bounding the Rademacher complexity of Fourier neural operators* (Machine Learning, 2024).
- **Status:** Published Springer Machine Learning article; Fig. 1 restates the canonical FNO architecture and layer.
- **URL:** https://link.springer.com/article/10.1007/s10994-024-06533-y
- **Actual Fourier-KAN?** No. This is Fourier Neural Operator (FNO), included as a transferable Fourier-operator visualization precedent rather than a KAN source.
- **Figure checked:** Fig. 1 has two panels: (a) overall architecture `input → lifting P → Fourier layers → projection Q → output`; (b) detailed Fourier layer with a spectral branch (FFT → learned frequency weights R → inverse FFT) and a local linear branch W, merged before activation.
- **Relevant evidence:** The article calls panel (a) a “Sketch of the overall architecture” and panel (b) a “Detailed diagram of the Fourier layers”; text says the layer sums the neural-network/local branch with the Fourier-kernel convolution branch.
- **What to borrow for T60 branch:** Borrow the clean two-panel hierarchy: a compact T60-head overview plus a separate Fourier operation detail panel. If the T60 head actually applies Fourier basis functions directly (rather than FFT), replace FFT/IFFT with sin/cos basis expansion while retaining the branch → merge → nonlinearity visual grammar.
- **Recommended detail level:** Use the overall-vs-detail split and branch merging, but do not imply that the T60 head is an FNO. Label the inset “Fourier basis expansion” and keep FNO only as a figure-design precedent.
- **Short quote (21 words):** “a Sketch of the overall architecture of Fourier neural operator (FNO) b Detailed diagram of the Fourier layers.”
