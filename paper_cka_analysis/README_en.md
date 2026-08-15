# Paper-grade CKA representation analysis

This directory reproduces the shared-layer representation similarity analysis between the CMGAN-T60 single-task model and the multi-task models. The analyzed layers are the Dense Encoder, TSCB-1, and TSCB-2. The primary model is the denoise-dominant configuration with α=1 and β=0.05.

## Experimental protocol

- Data: all 1,080 utterances from each of test1--test4 (4,320 utterances in total).
- Pairing: within each test set, both models always receive the same utterances in the same order.
- Representation: each `(C,T,F)` activation is globally averaged over time and frequency, producing one 64-dimensional vector per utterance.
- Metric: centered linear CKA. The implementation uses the feature-space expression that is mathematically equivalent to the Gram-matrix expression, avoiding explicit `N x N` matrices.
- Uncertainty: 1,000 paired bootstrap resamples are performed for every test-set/model/layer cell. Each resample draws N indices with replacement and applies the same indices to both models. The percentile 95% confidence interval is defined by the 2.5th and 97.5th percentiles.
- Cross-condition summary: the macro mean and sample standard deviation across the four test sets are reported.
- Sensitivity check: representations from all four test sets are concatenated to compute pooled CKA over 4,320 utterances. This is supplementary rather than the primary result.

## Outputs

After completion, `results/` contains:

- `cka_results.csv`: per-test-set estimates, macro mean/SD, and pooled CKA.
- `bootstrap_distributions.npz`: all bootstrap replicates for independent verification.
- `metadata.json`: data, checkpoints, seed, and statistical protocol.
- `features/*.npz`: cached features for auditing and resumable execution.
- `figures/fig_primary_b005_per_split_ci95.{png,pdf}`: layerwise CKA and 95% CIs for the β=0.05 primary model.
- `figures/fig_all_models_macro_mean_sd.{png,pdf}`: macro-mean CKA and between-test-set SD for all weighting configurations.
- `run.log`: complete execution log.

## Reproduction command

```bash
cd /mnt/tidal-sh01/usr/chuan/youling/project/11/multitask-experiments/kan-multitask
setsid nohup /mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python \
  paper_cka_analysis/run_paper_cka.py \
  --gpu 2 --bootstrap 1000 \
  > paper_cka_analysis/results/run.log 2>&1 < /dev/null &
```

Existing feature caches are reused by default. Use `--force_extract` only after changing a checkpoint or the feature-extraction procedure.

## Interpretation boundary

A lower CKA indicates that multi-task representations deviate more strongly from the single-task reference; it does not by itself demonstrate higher representation quality. CKA should be interpreted jointly with linear-probe results and final T60 metrics: CKA quantifies representational change, the probe quantifies linear T60 decodability, and RMSE/MAE quantify task performance.

All manuscript references to figures and tables must include identifiers, for example `Fig.~\ref{fig:cka_layerwise}` rather than “the figure below.”
