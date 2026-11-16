"""Compute six fixed-RIR case metrics from existing prediction archives only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SIM_DIR = ROOT / 'outputs/sim_rir_case_candidates_2026-10-28_v1'
REAL_DIR = ROOT / 'outputs/real_rir_case_baselines_2026-10-28_v1'
SIM_META = SIM_DIR / 'recommended_sample_metadata.csv'
SIM_SELECTION = SIM_DIR / 'selection_record.json'
REAL_PRED = REAL_DIR / 'plot_data.csv'
REAL_METRICS = REAL_DIR / 'summary_metrics.csv'
REAL_SELECTION = ROOT / 'outputs/real_rir_case_candidates_2026-10-27_v1/metadata_selection_record.json'
CASE_IDS = ['S1', 'S2', 'S3', 'R13', 'R47', 'R61']
EXPECTED_N = {'S1': 15, 'S2': 11, 'S3': 12, 'R13': 13, 'R47': 19, 'R61': 11}
EXPECTED_T60 = {'S1': Decimal('0.290'), 'S2': Decimal('0.694'), 'S3': Decimal('1.088'), 'R13': Decimal('0.303'), 'R47': Decimal('0.697'), 'R61': Decimal('1.133')}
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
MODELS = [
    ('berp', 'BERP', 'papercompare/berp_t60_v7_fixed/test_results'),
    ('noiseaware', 'noise-aware', 'papercompare/noiseaware/fix-v7dataset-test'),
    ('damtl', 'DAMTL', 'papercompare/damtl/test'),
    ('ours', 'Ours', 'runs/kan_multitask_norm_w_denoise_b005_seed42/test'),
]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_values(errors):
    assert errors and all(math.isfinite(e) for e in errors)
    n = len(errors)
    return {
        'n': n,
        'rmse_ms': math.sqrt(math.fsum(e * e for e in errors) / n),
        'mae_ms': math.fsum(abs(e) for e in errors) / n,
        'bias_ms': math.fsum(errors) / n,
    }


def main():
    archive_paths = {}
    for model, label, folder in MODELS:
        for split in ['test2', 'test4']:
            if model == 'noiseaware':
                archive_paths[model, split] = ROOT / folder / f'fix_normalized_run1_test_{split}' / 'detailed_results.csv'
            else:
                archive_paths[model, split] = ROOT / folder / f'evaluation_results_{split}.json'
    protected = [ROOT / f'docs/{name}' for name in ['main_zh.md', 'main.tex', 'main_bilingual.md']]
    input_paths = [SIM_META, SIM_SELECTION, REAL_PRED, REAL_METRICS, REAL_SELECTION, *archive_paths.values(), *protected]
    before = {str(p): sha256(p) for p in input_paths}

    sim_selection = json.loads(SIM_SELECTION.read_text())
    sim_cases = {r['candidate_case_id']: r for r in sim_selection['primary_recommendations']}
    real_cases = {r['rir_id']: r for r in json.loads(REAL_SELECTION.read_text())['rows'] if r['rir_id'] in EXPECTED_N}
    samples = []
    for row in read_csv(SIM_META):
        case = row['candidate_case_id']
        assert row['content_sha256'] == sim_cases[case]['content_sha256']
        assert row['original_filename'] == sim_cases[case]['original_filename']
        samples.append({
            'case_id': case, 'rir_type': 'simulated', 'source_dataset': 'ACE_sim',
            'scene_or_source_group': row['source_room_token'], 'split': row['split'],
            'sample_name': row['sample_name'], 'historical_t60_s': row['historical_t60_s'],
            'snr_db': row['snr_db'], 'content_sha256': row['content_sha256'],
            'original_filename': row['original_filename'],
        })
    real_prior_rows = read_csv(REAL_PRED)
    for row in real_prior_rows:
        if row['model'] != 'ours':
            continue
        case = row['case_id']
        assert row['content_sha256'] == real_cases[case]['sha256']
        assert row['original_filename'] == real_cases[case]['original_filename']
        samples.append({
            'case_id': case, 'rir_type': 'real', 'source_dataset': 'AIR',
            'scene_or_source_group': row['scene'], 'split': row['split'],
            'sample_name': row['sample_name'], 'historical_t60_s': row['historical_t60_s'],
            'snr_db': row['snr_db'], 'content_sha256': row['content_sha256'],
            'original_filename': row['original_filename'],
        })
    assert len(samples) == 81
    assert len({(r['split'], r['sample_name']) for r in samples}) == 81
    assert Counter(r['case_id'] for r in samples) == EXPECTED_N
    assert Counter(r['rir_type'] for r in samples) == {'simulated': 38, 'real': 43}
    assert len({r['content_sha256'] for r in samples}) == 6
    for row in samples:
        assert Decimal(row['historical_t60_s']) == EXPECTED_T60[row['case_id']]
        assert row['split'] in ['test2', 'test4']
    samples.sort(key=lambda r: (CASE_IDS.index(r['case_id']), r['split'], r['sample_name']))

    indices = {}
    archive_metadata = []
    for (model, split), path in archive_paths.items():
        if model == 'noiseaware':
            rows = [{
                'sample_name': r['Sample_Name'], 'true_t60': r['T60_True'],
                'pred_t60': r['T60_Pred'], 'snr': r['SNR_dB'],
            } for r in read_csv(path)]
            metadata = {'format': 'detailed_results.csv'}
        else:
            data = json.loads(path.read_text())
            assert data['test_split'] == split
            rows = data['per_sample_results']
            metadata = {k: data.get(k) for k in ['experiment_name', 'dataset_version', 'test_split']}
        assert len(rows) == 1080
        indices[model, split] = {r['sample_name']: r for r in rows}
        assert len(indices[model, split]) == len(rows)
        archive_metadata.append({'model': model, 'split': split, 'path': str(path), 'n_source_samples': len(rows), **metadata})

    joined = []
    for sample in samples:
        for model, label, _ in MODELS:
            row = indices[model, sample['split']][sample['sample_name']]
            true_s = Decimal(str(row['true_t60']))
            pred_s = Decimal(str(row['pred_t60']))
            assert true_s == Decimal(sample['historical_t60_s'])
            assert Decimal(str(row['snr'])) == Decimal(sample['snr_db'])
            assert true_s.is_finite() and pred_s.is_finite()
            error_ms = float((pred_s - true_s) * 1000)
            joined.append({
                **sample, 'model': model, 'model_label': label,
                'prediction_t60_s': str(pred_s), 'signed_error_ms': error_ms,
                'prediction_archive_path': str(archive_paths[model, sample['split']]),
                'join_status': 'ok',
            })
    assert len(joined) == 324
    assert len({(r['split'], r['sample_name'], r['model']) for r in joined}) == 324
    prior_index = {(r['split'], r['sample_name'], r['model']): r for r in real_prior_rows}
    prior_metric_index = {(r['case_id'], r['model']): r for r in read_csv(REAL_METRICS)}
    for row in joined:
        if row['rir_type'] == 'real':
            previous = prior_index[row['split'], row['sample_name'], row['model']]
            assert Decimal(row['prediction_t60_s']) == Decimal(previous['prediction_t60_s'])
            assert abs(row['signed_error_ms'] - float(previous['signed_error_ms'])) < 1e-9

    case_metrics = []
    coverage_rows = []
    for case in CASE_IDS:
        case_samples = [r for r in samples if r['case_id'] == case]
        snr_counts = Counter(int(float(r['snr_db'])) for r in case_samples)
        split_counts = Counter(r['split'] for r in case_samples)
        first = case_samples[0]
        coverage_rows.append({
            'case_id': case, 'rir_type': first['rir_type'], 'source_dataset': first['source_dataset'],
            'scene_or_source_group': first['scene_or_source_group'], 'historical_t60_s': str(EXPECTED_T60[case]),
            'n_samples': len(case_samples), 'n_test2': split_counts['test2'], 'n_test4': split_counts['test4'],
            **{f'n_snr_{snr}db': snr_counts[snr] for snr in SNR_LEVELS},
            'content_sha256': first['content_sha256'], 'original_filename': first['original_filename'],
        })
        for model, label, _ in MODELS:
            errors = [r['signed_error_ms'] for r in joined if r['case_id'] == case and r['model'] == model]
            metrics = metric_values(errors)
            if first['rir_type'] == 'real':
                previous = prior_metric_index[case, model]
                for metric in ['rmse_ms', 'mae_ms', 'bias_ms']:
                    assert abs(metrics[metric] - float(previous[metric])) < 1e-9
            case_metrics.append({
                'case_id': case, 'rir_type': first['rir_type'], 'historical_t60_s': str(EXPECTED_T60[case]),
                'model': model, 'model_label': label, **metrics,
            })

    pooled_metrics = []
    for scope, rir_type, expected_n in [('simulated_38', 'simulated', 38), ('real_43', 'real', 43), ('all_81', None, 81)]:
        for model, label, _ in MODELS:
            errors = [r['signed_error_ms'] for r in joined if r['model'] == model and (rir_type is None or r['rir_type'] == rir_type)]
            assert len(errors) == expected_n
            pooled_metrics.append({'scope': scope, 'model': model, 'model_label': label, **metric_values(errors)})

    write_csv(OUT / 'case_metrics.csv', case_metrics)
    write_csv(OUT / 'pooled_metrics.csv', pooled_metrics)
    write_csv(OUT / 'sample_predictions_joined.csv', joined)
    write_csv(OUT / 'sample_coverage.csv', coverage_rows)
    assert all(sha256(Path(path)) == value for path, value in before.items())
    record = {
        'selection': 'S1/S2/S3 explicitly adopted by the user, together with the fixed R13/R47/R61; no case substitution or sample filtering.',
        'n_speech_samples': 81, 'n_simulated_speech_samples': 38, 'n_real_speech_samples': 43,
        'n_models': 4, 'n_predictions': 324, 'n_case_model_groups': 24,
        'models': {model: label for model, label, _ in MODELS}, 'case_counts': EXPECTED_N,
        'speech_sample_counts_by_split': dict(Counter(r['split'] for r in samples)),
        'all_predictions_joined': True, 'missing_predictions': 0, 'duplicate_selected_keys': 0,
        'reference_or_snr_conflicts': 0, 'nonfinite_predictions': 0,
        'real_43_predictions_and_metrics_match_previous_output': True,
        'reference': 'Unchanged historical T60 labels from the fixed sample metadata; never replaced by filename rt tokens.',
        'error_ms': '1000 * (saved_prediction_s - historical_reference_s)',
        'rmse_ms': 'sqrt(sum(error_ms**2)/n)', 'mae_ms': 'sum(abs(error_ms))/n',
        'bias_ms': 'sum(error_ms)/n; positive means overestimation, negative underestimation',
        'pooled_metrics': 'Combine all sample errors within each stated scope. Do not average case RMSEs.',
        'numerics': 'Decimal used for exact label/SNR comparisons and subtraction of saved decimal predictions; float math.fsum used for means and squared sums.',
        'archive_error_ms_fields_used': False, 'archives': archive_metadata,
        'input_sha256': before, 'inputs_and_manuscripts_unchanged': True,
        'plots_created': False, 'models_loaded': False, 'inference_training_audio_processing_or_label_recalculation': False,
        'limitations': [
            'Existing prediction archives have saved precision limitations, commonly four decimal places in seconds.',
            'Six cases have different speech, SNR and measurement configurations. Aggregates are descriptive, not matched real/simulated comparisons.',
            'Existing provenance and migration/cropping limitations remain unchanged; no new audio or RIR identity checks were performed.',
        ],
    }
    (OUT / 'calculation_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'case_metrics': case_metrics, 'pooled_metrics': pooled_metrics, 'validation': {'speech_samples': 81, 'predictions': 324, 'all_joins_ok': True, 'previous_real_metrics_match': True, 'inputs_and_manuscripts_unchanged': True, 'plots_created': False}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
