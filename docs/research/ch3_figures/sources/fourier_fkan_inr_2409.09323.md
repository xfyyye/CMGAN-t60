# Fourier-KAN/FKAN — learnable Fourier activation block

- **Paper:** Ali Mehrabian et al., *Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks*.
- **Status:** arXiv:2409.09323; arXiv page states accepted for IEEE ICASSP 2025 (v3, 14 Jan 2025).
- **URL:** https://arxiv.org/abs/2409.09323 (PDF: https://arxiv.org/pdf/2409.09323)
- **Actual Fourier-KAN?** Yes. The first layer uses learnable edge activations parameterized by Fourier-series coefficients; later layers are linear + tanh.
- **Figure checked:** Fig. 1, caption: “Illustration of the proposed FKAN model.” It depicts an FKAN block for task-specific frequency components, followed by L hidden layers and an output projection. The body text explicitly says Fig. 1 shows Fourier-series parameterization of each learnable function.
- **Relevant evidence:** The paper defines each edge function as `psi(x)=sum_k(a_k sin(kx)+b_k cos(kx))`, with `a_k,b_k` learnable and `K` the frequency/grid-size hyperparameter. It then stacks fixed nonlinear hidden layers and a final linear output.
- **What to borrow for T60 branch:** Use a compact inset for one Fourier-KAN edge/function: input scalar → parallel sin(kx), cos(kx) basis terms → learnable coefficient weighting → summation, then show the resulting FKAN block feeding hidden layers. Do not draw every neuron; the paper’s figure keeps the Fourier detail localized to the first block.
- **Recommended detail level:** Show basis family and coefficient aggregation once, and represent repeated hidden layers as a single bracket/stack. This is the clearest precedent for explaining why the T60 head is Fourier-based without overloading the full network diagram.
- **Short quote (20 words):** “The proposed architecture includes an FKAN block for capturing task-specific frequency components with learnable activation functions.”
