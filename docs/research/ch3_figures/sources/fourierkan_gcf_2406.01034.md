# FourierKAN-GCF — overview with fitted Fourier curve inset

- **Paper:** Jinfeng Xu et al., *Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation* (original title/version: FourierKAN-GCF).
- **Status:** arXiv:2406.01034 v3; PDF identifies publication in ACM CIKM 2025 (short paper).
- **URL:** https://arxiv.org/abs/2406.01034 (PDF: https://arxiv.org/pdf/2406.01034)
- **Actual Fourier-KAN?** Yes. Fourier KAN replaces the MLP feature transformation; learnable activation functions live on edges.
- **Figure checked:** Fig. 1, “Overview of FourierKAN-GCF.” It combines (i) a high-level multi-layer graph propagation overview, (ii) a zoomed FourierKAN edge/curve area, and (iii) the explicit Fourier expansion formula. The PDF text labels the inset “Fitted Fourier curve on edges” and explains that each edge has a learnable Fourier curve.
- **Relevant evidence:** Figure 1 includes repeated “Normalized Sum * Fourier KAN” blocks, then a small edge-level inset with “each learnable Fourier curve,” “Fitted Fourier curve on edges,” and the sine/cosine coefficient expression.
- **What to borrow for T60 branch:** Adopt a two-scale figure: main T60 head flow (shared feature → FourierKAN layer(s) → scalar T60 output) plus one enlarged edge-level Fourier curve inset. Put the Fourier formula next to the inset, rather than expanding all connections.
- **Recommended detail level:** This is the best precedent when the paper needs both an architecture overview and a visual explanation of per-edge Fourier functions. Keep only one representative edge and one coefficient aggregation formula.
- **Short quote (19 words):** “Figure 1: Overview of FourierKAN-GCF.”
