# DPT-FSNet (Yu et al., 2021) — multi-panel detail convention

- Poster/PDF: [DPT-FSNet: Dual-path Transformer Based Full-band and Sub-band Fusion Network](https://sigport.org/sites/default/files/docs/Poster_2333.pdf)
- **Figure 1(a–d)** uses a compact overview plus separate enlarged panels for the dense block, dual-path transformer, and decoder. This is a strong precedent for the requested “concept figure → module-detail figures” organization.
- The decoder panel is treated as an output head that estimates a complex ratio mask; the dense block and dual-path module are expanded independently. The key lesson is to avoid putting all internals in the overview figure and instead use matching panel labels/callouts.

## Reusable choices

- Number each detail panel and use the same block name/color in the overview and detail figure.
- Use a dashed outline around a module whose internal repetition is shown in its own detail panel.
- Keep module detail figures self-contained, with one input and one output arrow, so they can be read independently in the paper.

