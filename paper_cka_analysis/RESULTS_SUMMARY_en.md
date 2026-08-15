# CKA analysis results summary

## Primary result: the β=0.05 model versus the single-task reference

Table 1 reports centered linear CKA over all utterances in each test set. Brackets contain percentile 95% confidence intervals from 1,000 paired bootstrap resamples.

**Table 1. Layerwise CKA between the single-task and β=0.05 multi-task models**

| Test set | Dense Encoder | TSCB-1 | TSCB-2 |
|---|---:|---:|---:|
| test1 | 0.9501 [0.9414, 0.9579] | 0.6791 [0.6547, 0.7063] | 0.4729 [0.4543, 0.4980] |
| test2 | 0.9543 [0.9466, 0.9610] | 0.6723 [0.6400, 0.7063] | 0.4948 [0.4711, 0.5232] |
| test3 | 0.9540 [0.9471, 0.9601] | 0.7217 [0.7005, 0.7434] | 0.5305 [0.5095, 0.5566] |
| test4 | 0.9470 [0.9390, 0.9538] | 0.6921 [0.6688, 0.7163] | 0.5543 [0.5318, 0.5811] |
| Macro mean ± SD | 0.9514 ± 0.0035 | 0.6913 ± 0.0219 | 0.5132 ± 0.0363 |

All four test sets exhibit the same layerwise pattern: the Dense Encoder remains highly similar to the single-task reference, CKA decreases markedly at TSCB-1, and decreases further at TSCB-2. This indicates that representational changes induced by β=0.05 multi-task supervision accumulate primarily in the shared Conformer blocks rather than in the low-level encoder. The consistency across test sets shows that the observation is not restricted to one RIR or noise condition.

## Comparison across weighting configurations

**Table 2. Macro-mean CKA across the four test sets (mean ± between-test-set SD)**

| Model | Dense Encoder | TSCB-1 | TSCB-2 |
|---|---:|---:|---:|
| β=0.05, denoise-dominant | 0.9514 ± 0.0035 | 0.6913 ± 0.0219 | 0.5132 ± 0.0363 |
| α=1, β=1, balanced | 0.9874 ± 0.0012 | 0.7322 ± 0.0140 | 0.7023 ± 0.0197 |
| α=0.1, β=1, T60-dominant | 0.9961 ± 0.0008 | 0.7687 ± 0.0161 | 0.8140 ± 0.0026 |

The β=0.05 model is least similar to the single-task model at TSCB-2, the balanced model is intermediate, and the T60-dominant model is most similar. This ordering is consistent with the expected effect of loss weighting on shared representations, although CKA alone does not establish that lower similarity implies higher representation quality.

## Pooled sensitivity check

After concatenating all 4,320 representations from test1--test4, pooled CKA for the β=0.05 model is 0.9518, 0.6823, and 0.4992 at the Dense Encoder, TSCB-1, and TSCB-2, respectively. The pooled result preserves the layerwise pattern of the per-test-set macro summary, showing that the main conclusion is not sensitive to the aggregation strategy.

## Suggested manuscript wording

> As shown in Fig. X, representation similarity between the β=0.05 multi-task model and the single-task reference progressively decreases from the Dense Encoder to TSCB-2. Dense Encoder CKA remains approximately 0.95 across all four test sets, whereas TSCB-2 CKA ranges from 0.47 to 0.55, with paired-bootstrap 95% confidence intervals supporting the same layerwise pattern. These results indicate that denoise-dominant joint supervision predominantly reshapes higher-level representations in the shared Conformer blocks while leaving low-level encoding relatively stable.

The interpretation should be accompanied by T60 performance and linear-probe evidence. The defensible claim is that the representation changes more strongly relative to the single-task reference; lower CKA alone does not prove higher representation quality or causality.

Full-precision estimates are available in `results/cka_results.csv`, and raw bootstrap distributions are stored in `results/bootstrap_distributions.npz`.
