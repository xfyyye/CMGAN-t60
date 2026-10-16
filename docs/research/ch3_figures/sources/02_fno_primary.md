# Fourier Neural Operator (primary architecture reference)
URL: https://arxiv.org/abs/2010.08895

Figure 1 gives both the full architecture and a Fourier-layer detail. The full path is lifting P -> repeated Fourier layers -> projection Q; the detail expands a Fourier transform F -> learned mode-wise linear map R on retained modes -> inverse transform F^-1, in parallel with a local linear W, followed by addition and activation.

Useful convention: for the paper's T60 branch, show a compact Fourier basis expansion rather than a generic “KAN” box: input feature -> Fourier basis bank (sin/cos) -> learned coefficients/weights -> summation -> regression output. If the implementation is a finite FourierKAN activation rather than an FFT spectral convolution, retain the visual grammar (basis bank + coefficients + aggregation) but do not label it FFT/FNO.

Evidence excerpt (source text): Figure 1 is described as “the architecture of the Fourier layer”; the text states that the top path applies F, R on lower modes, and F^-1, while the bottom path applies W.
