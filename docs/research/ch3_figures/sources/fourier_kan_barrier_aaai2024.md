# Fourier KAN barrier certificates — explicit edge-function network

- **Paper:** Chao et al., *Formal Synthesis of Barrier Certificates Using Fourier Kolmogorov-Arnold Network*.
- **Status:** AAAI conference proceedings paper (AAAI 2025 proceedings PDF; DOI/venue metadata in PDF). This is a published paper, not only a preprint.
- **URL:** https://ojs.aaai.org/index.php/AAAI/article/download/32101/34256
- **Actual Fourier-KAN?** Yes. It replaces B-spline activations with explicit Fourier coefficient functions.
- **Figure checked:** Fig. 1 (PDF p. 2/3), caption: “Left: A 2-Layer Fourier KAN with shape [2, 2, 1]. Right: an activation function is parameterized as a Fourier Curve.” The left panel is a node/edge network; the right panel isolates one learnable Fourier activation curve.
- **Relevant evidence:** The text states there are `n_l n_{l+1}` activation functions between adjacent layers and gives the edge sum of sine/cosine terms. It works through a concrete two-input example where two Fourier activations are summed into one hidden node.
- **What to borrow for T60 branch:** Use this as the strongest precedent for an explanatory inset: (a) simplified layer connectivity, (b) one edge activation expanded into a Fourier curve, and (c) summation into the next neuron. It makes the “functions on edges” concept immediately readable.
- **Recommended detail level:** For your scalar T60 regression head, show one representative input-to-hidden edge with the Fourier curve/basis expansion and collapse the rest of the layer into repeated edge blocks. Avoid plotting a learned curve unless you intend to show actual trained coefficients.
- **Short quote (21 words):** “Left: A 2-Layer Fourier KAN with shape [2, 2, 1]. Right: an activation function is parameterized as a Fourier Curve.”
