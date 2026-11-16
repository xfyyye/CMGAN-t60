"""Four-Test pooled RMSE by actual RIR type and 0.1-second historical T60 bin."""

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_MAP = Path('/Users/xiexinni/case study/four_pool_content_audit_2026-10-27_v1/sample_mapping.csv')
SPLITS = ['test1', 'test2', 'test3', 'test4']
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
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {(m, s): ROOT / folder / (f'fix_normalized_run1_test_{s}/detailed_results.csv' if m == 'noiseaware' else f'evaluation_results_{s}.json') for m, _, folder in MODELS for s in SPLITS}
    protected = [ROOT / f'docs/{name}' for name in ['main_zh.md', 'main.tex', 'main_bilingual.md']]
    inputs = [SOURCE_MAP, *paths.values(), *protected]
    hashes = {str(p): digest(p) for p in inputs}
    mapping = [r for r in read_csv(SOURCE_MAP) if r['split'] in SPLITS]
    map_index = {(r['split'], int(r['sample_id'])): r for r in mapping}
    assert len(mapping) == len(map_index) == 4320
    assert Counter(r['source_type'] for r in mapping) == {'real': 462, 'simulated': 3858}
    rir_types, rir_labels = defaultdict(set), defaultdict(set)
    for row in mapping:
        assert row['sha256']
        rir_types[row['sha256']].add(row['source_type'])
        rir_labels[row['sha256']].add(Decimal(row['manifest_t60']))
    assert len(rir_types) == 647 and all(len(v) == 1 for v in rir_types.values())
    assert all(len(v) == 1 for v in rir_labels.values())

    indices, archive_metadata = {}, []
    for (model, split), path in paths.items():
        if model == 'noiseaware':
            rows = [{'sample_name': r['Sample_Name'], 'true_t60': r['T60_True'], 'pred_t60': r['T60_Pred'], 'snr': r['SNR_dB']} for r in read_csv(path)]
            metadata = {'format': 'detailed_results.csv'}
        else:
            data = json.loads(path.read_text())
            assert data['test_split'] == split
            rows = data['per_sample_results']
            metadata = {k: data.get(k) for k in ['experiment_name', 'dataset_version', 'test_split']}
        indices[model, split] = {r['sample_name']: r for r in rows}
        assert len(rows) == len(indices[model, split]) == 1080
        archive_metadata.append({'model': model, 'split': split, 'path': str(path), 'n_samples': len(rows), **metadata})

    joined = []
    for split in SPLITS:
        canonical = indices['ours', split]
        for model, _, _ in MODELS:
            assert set(indices[model, split]) == set(canonical)
        for name, reference_row in canonical.items():
            match = re.fullmatch(r'speech(\d+)_reverb_[-\d.]+_[-\d.]+dB', name)
            assert match
            sample_id = int(match.group(1))
            source = map_index[split, sample_id]
            reference_s = Decimal(source['manifest_t60'])
            snr = Decimal(str(reference_row['snr']))
            assert Decimal('0.1') <= reference_s <= Decimal('1.5')
            bin_index = min(int(reference_s * 10) - 1, 13)
            lo, hi = Decimal(bin_index + 1) / 10, Decimal(bin_index + 2) / 10
            assert lo <= reference_s and (reference_s < hi or hi == Decimal('1.5'))
            row = {
                'split': split, 'sample_id': sample_id, 'sample_name': name,
                'rir_type': source['source_type'], 'content_sha256': source['sha256'],
                'historical_t60_s': str(reference_s), 'snr_db': str(snr),
                'bin_index': bin_index, 'bin_low_s': str(lo), 'bin_high_s': str(hi),
                'historical_rir_path': source['historical_rir_path'],
            }
            for model, _, _ in MODELS:
                prediction = indices[model, split][name]
                true_s = Decimal(str(prediction['true_t60']))
                pred_s = Decimal(str(prediction['pred_t60']))
                assert true_s == reference_s
                assert Decimal(str(prediction['snr'])) == snr
                assert pred_s.is_finite() and true_s.is_finite()
                row[f'{model}_prediction_s'] = str(pred_s)
                row[f'{model}_error_ms'] = float((pred_s - reference_s) * 1000)
            joined.append(row)
    assert len(joined) == 4320
    assert len({(r['split'], r['sample_name']) for r in joined}) == 4320
    assert len({(r['split'], r['sample_id']) for r in joined}) == 4320

    def summarize(rows, kind, label, low=None, high=None, final=False):
        result = {'rir_type': kind, 't60_bin_s': label, 'bin_low_s': low, 'bin_high_s': high, 'right_inclusive': final,
                  'rir_count': len({r['content_sha256'] for r in rows}), 'speech_count': len(rows)}
        for model, _, _ in MODELS:
            sse = math.fsum(r[f'{model}_error_ms'] ** 2 for r in rows)
            result[f'{model}_squared_error_sum_ms2'] = sse
            result[f'{model}_rmse_ms'] = math.sqrt(sse / len(rows)) if rows else None
        return result

    results = {}
    overall = []
    for kind, expected_n, expected_rirs, name in [('real', 462, 61, '实测'), ('simulated', 3858, 586, '仿真')]:
        subset = [r for r in joined if r['rir_type'] == kind]
        bins = []
        for index in range(14):
            low, high = (index + 1) / 10, (index + 2) / 10
            label = f'[{low:.1f}, {high:.1f}' + (']' if index == 13 else ')')
            group = [r for r in subset if r['bin_index'] == index]
            bins.append(summarize(group, kind, label, low, high, index == 13))
        assert sum(r['speech_count'] for r in bins) == expected_n
        assert sum(r['rir_count'] for r in bins) == expected_rirs
        total = summarize(subset, kind, f'全部{name}')
        assert total['rir_count'] == expected_rirs
        results[kind] = bins + [total]
        overall.append(total)
        report_fields = ['t60_bin_s', 'rir_count', 'speech_count', 'berp_rmse_ms', 'noiseaware_rmse_ms', 'damtl_rmse_ms', 'ours_rmse_ms']
        write_csv(OUT / f'{kind}_t60_bin_rmse.csv', [{k: r[k] for k in report_fields} for r in results[kind]])
    overall.append(summarize(joined, 'all', '四Test全部'))
    write_csv(OUT / 'overall_rmse.csv', overall)
    write_csv(OUT / 'sample_predictions_joined.csv', joined)
    assert all(digest(Path(path)) == value for path, value in hashes.items())
    result = {
        'tables': results, 'overall': overall,
        'models': [{'key': model, 'label': label} for model, label, _ in MODELS],
        'n_samples': 4320, 'n_predictions': 17280, 'n_unique_rirs': 647,
        'classification': 'Existing RIR content attribution, not nominal Test or pool names.',
        'scope': 'Pool Test1-Test4 speech samples within each actual RIR type and historical T60 bin.',
        'binning': 'Decimal boundaries: [0.1,0.2), ..., [1.4,1.5]. Last bin includes 1.5.',
        'rir_counts': 'SHA256 content deduplication across all four Tests. Each RIR has one consistent source type and historical label.',
        'rmse': 'sqrt(sum((1000*(saved_prediction_s - historical_reference_s))**2)/n_speech_samples)',
        'weighting': 'One weight per speech sample; no RIR-equal weighting, no SNR balancing, no averaging of Test/bin RMSE values.',
        'empty_bins': 'Counts zero; RMSE null, never zero.',
        'archive_error_fields_used': False, 'archives': archive_metadata,
        'join_checks': {'missing': 0, 'duplicates': 0, 'label_conflicts': 0, 'snr_conflicts': 0, 'nonfinite_predictions': 0, 'unbinned_samples': 0},
        'input_sha256': hashes, 'inputs_and_manuscripts_unchanged': True,
        'new_inference_training_audio_or_label_processing': False, 'plots_created': False,
        'limitations': [
            'Saved prediction precision applies, commonly four decimal places in seconds.',
            'This pooled metric differs from the paper arithmetic mean of four Test RMSEs.',
            'Types and bins have different T60/SNR/source distributions. These tables do not isolate causal RIR-type or T60 effects.',
            'RIR identity is content-level. Multiple speech samples sharing one RIR are not independent rooms.',
            'Existing historical archive/provenance and migration/cropping limitations remain unchanged.',
        ],
    }
    (OUT / 'calculation_record.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'tables': results, 'overall': overall, 'n_predictions': 17280, 'inputs_and_manuscripts_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
