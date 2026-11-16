"""Recommend simulated-RIR cases using provenance and coverage only."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIT = Path('/Users/xiexinni/case study/four_pool_content_audit_2026-10-27_v1')
SAMPLE_MAP = AUDIT / 'sample_mapping.csv'
RIR_MAP = AUDIT / 'rir_mapping.csv'
ACE_RENAME = Path('/Users/xiexinni/dataset/rawdataset/RIR/ir-ACE-simulated_1068_train_rirsimulated/t60_calculation_results.csv')
REAL_SELECTION = ROOT / 'outputs/real_rir_case_candidates_2026-10-27_v1/metadata_selection_record.json'
SNR = [-5, 0, 5, 10, 15, 20]
TARGETS = [('S1', 'short', 'R13', Decimal('0.303')), ('S2', 'medium', 'R47', Decimal('0.697')), ('S3', 'long', 'R61', Decimal('1.133'))]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archives = {s: ROOT / 'runs/kan_multitask_norm_w_denoise_b005_seed42/test' / f'evaluation_results_{s}.json' for s in ['test1', 'test2', 'test3', 'test4']}
    protected = [ROOT / f'docs/{name}' for name in ['main_zh.md', 'main.tex', 'main_bilingual.md']]
    paths = [SAMPLE_MAP, RIR_MAP, ACE_RENAME, REAL_SELECTION, *archives.values(), *protected]
    before = {str(p): sha(p) for p in paths}
    # Extract only the three metadata fields needed for coverage. Prediction and error fields are never read.
    metadata = {}
    for split, path in archives.items():
        data = json.loads(path.read_text())
        assert data['test_split'] == split and len(data['per_sample_results']) == 1080
        for row in data['per_sample_results']:
            match = re.fullmatch(r'speech(\d+)_reverb_[-\d.]+_[-\d.]+dB', row['sample_name'])
            assert match, row['sample_name']
            key = split, int(match.group(1))
            assert key not in metadata
            metadata[key] = {'sample_name': row['sample_name'], 'snr_db': int(float(row['snr'])), 'historical_t60_s': str(row['true_t60'])}
    mappings = read_csv(RIR_MAP)
    by_hash = defaultdict(list)
    for row in mappings:
        by_hash[row['sha256']].append(row)
    rename_rows = [r for r in read_csv(ACE_RENAME) if r['success'] == 'True']
    rename = {r['new_filename']: r['original_filename'] for r in rename_rows}
    assert len(rename) == len(rename_rows)
    groups = defaultdict(list)
    for row in read_csv(SAMPLE_MAP):
        if row['split'] not in archives or row['source_type'] != 'simulated':
            continue
        meta = metadata[row['split'], int(row['sample_id'])]
        assert Decimal(row['manifest_t60']) == Decimal(meta['historical_t60_s'])
        groups[row['sha256']].append({**row, **meta})
    coverage = []
    for content_hash, samples in groups.items():
        pool_rows = by_hash[content_hash]
        assert all(r['source_type'] == 'simulated' for r in pool_rows)
        evidence = [e for r in pool_rows for e in json.loads(r['evidence'])]
        source_datasets = sorted({e['source_dataset'] for e in evidence})
        assert len(source_datasets) == 1
        labels = {Decimal(r['manifest_t60']) for r in samples}
        assert len(labels) == 1
        original = ''
        if source_datasets == ['ACE_sim']:
            originals = {rename[Path(e['rir_path']).name] for e in evidence if Path(e['rir_path']).name in rename}
            assert len(originals) == 1
            original = next(iter(originals))
        test_counts = Counter(s['split'] for s in samples)
        snr = Counter(s['snr_db'] for s in samples)
        snr24 = Counter(s['snr_db'] for s in samples if s['split'] in ['test2', 'test4'])
        room_match = re.match(r'h_(room\d+)_', original)
        coverage.append({
            'content_sha256': content_hash, 'source_dataset': source_datasets[0],
            'historical_t60_s': float(next(iter(labels))), 'original_filename': original,
            'source_room_token': room_match.group(1) if room_match else '',
            'final_rir_filename': Path(samples[0]['historical_rir_path']).name,
            'n_test1': test_counts['test1'], 'n_test2': test_counts['test2'],
            'n_test3': test_counts['test3'], 'n_test4': test_counts['test4'],
            'n_all_tests': len(samples), 'n_test2_test4': test_counts['test2'] + test_counts['test4'],
            'snr_counts_all_tests': json.dumps([snr[x] for x in SNR]),
            'snr_counts_test2_test4': json.dumps([snr24[x] for x in SNR]),
            'snr_levels_test2_test4': len(snr24),
            'historical_rir_paths': json.dumps(sorted({s['historical_rir_path'] for s in samples})),
            'content_hit_paths': json.dumps(sorted({p for r in pool_rows for p in json.loads(r['all_hit_paths'])})),
        })
    coverage.sort(key=lambda r: (r['historical_t60_s'], r['content_sha256']))
    assert len(coverage) == 586 and sum(r['n_all_tests'] for r in coverage) == 3858
    assert Counter(r['source_dataset'] for r in coverage) == {'ACE_sim': 402, 'OpenSLR_sim': 184}
    scope24 = [r for r in coverage if r['n_test2_test4'] > 0]
    assert len(scope24) == 162 and sum(r['n_test2_test4'] for r in scope24) == 1913
    assert Counter(r['source_dataset'] for r in scope24) == {'ACE_sim': 102, 'OpenSLR_sim': 60}
    # This is a practical metadata coverage rule, not a statistical power claim.
    eligible = [r for r in scope24 if r['source_dataset'] == 'ACE_sim' and r['original_filename'] and r['n_test2'] > 0 and r['n_test4'] > 0 and r['n_test2_test4'] >= 10 and r['snr_levels_test2_test4'] >= 5]
    shortlist, selected, sample_rows = [], [], []
    for case_id, band, real_id, target in TARGETS:
        ranked = sorted(eligible, key=lambda r: (abs(Decimal(str(r['historical_t60_s'])) - target), -r['snr_levels_test2_test4'], -r['n_test2_test4'], r['content_sha256']))
        for rank, candidate in enumerate(ranked[:3], 1):
            item = {
                'candidate_case_id': case_id, 'band': band, 'rank': rank,
                'status': 'recommended' if rank == 1 else 'metadata_alternative',
                'corresponding_real_case_id': real_id, 'real_historical_t60_s': float(target),
                't60_distance_ms': float(abs(Decimal(str(candidate['historical_t60_s'])) - target) * 1000),
                **candidate,
            }
            shortlist.append(item)
            if rank == 1:
                selected.append(item)
                for sample in groups[candidate['content_sha256']]:
                    if sample['split'] not in ['test2', 'test4']:
                        continue
                    sample_rows.append({
                        'candidate_case_id': case_id, 'band': band, 'split': sample['split'],
                        'sample_id': sample['sample_id'], 'sample_name': sample['sample_name'],
                        'historical_t60_s': sample['historical_t60_s'], 'snr_db': sample['snr_db'],
                        'content_sha256': candidate['content_sha256'], 'source_dataset': candidate['source_dataset'],
                        'original_filename': candidate['original_filename'], 'source_room_token': candidate['source_room_token'],
                        'historical_rir_path': sample['historical_rir_path'],
                        'local_rir_path_from_server_record': sample['local_rir_path'],
                        'manifest_line': sample['manifest_line'],
                    })
    assert [r['historical_t60_s'] for r in selected] == [0.290, 0.694, 1.088]
    assert [r['n_test2_test4'] for r in selected] == [15, 11, 12]
    assert len(sample_rows) == 38 and len({(r['split'], r['sample_name']) for r in sample_rows}) == 38
    assert len({r['source_room_token'] for r in selected}) == 3
    write_csv(OUT / 'all_sim_rir_metadata_coverage.csv', coverage)
    write_csv(OUT / 'candidate_shortlist.csv', shortlist)
    write_csv(OUT / 'recommended_sample_metadata.csv', sample_rows)
    assert all(sha(Path(p)) == h for p, h in before.items())
    record = {
        'status': 'metadata_only_recommendations_not_performance_selected',
        'primary_recommendations': selected, 'metadata_alternatives': [r for r in shortlist if r['rank'] > 1],
        'eligible_candidate_count': len(eligible), 'snr_order_db': SNR,
        'selection_rules_in_order': [
            'Already attributed simulated RIR actually used in the existing four Tests; no new material or synthesis.',
            'Prefer Test2/Test4, the same split scope as the three fixed measured-RIR cases.',
            'Within that scope, ACE_sim provenance with a resolved original filename; both Test2 and Test4 represented.',
            'At least 10 speech samples and at least five of the six SNR levels. This is a practical coverage threshold, not statistical power.',
            'Minimize absolute historical T60 distance to fixed real cases 0.303/0.697/1.133 s; tie-break by SNR coverage, sample count, then hash.',
        ],
        'metadata_fields_extracted_from_existing_prediction_archives': ['sample_name', 'true_t60', 'snr'],
        'prediction_values_or_error_columns_used': False, 'model_metrics_computed': False,
        'no_claim_of_blinded_preregistration': True,
        'n_sim_rirs_all_tests': 586, 'n_sim_samples_all_tests': 3858,
        'sim_rir_sources_all_tests': {'ACE_sim': 402, 'OpenSLR_sim': 184},
        'n_sim_rirs_test2_test4': 162, 'n_sim_samples_test2_test4': 1913,
        'sim_rir_sources_test2_test4': {'ACE_sim': 102, 'OpenSLR_sim': 60},
        'n_recommended_sim_speech_samples': 38, 'n_existing_real_speech_samples': 43,
        'n_total_six_case_speech_samples_if_adopted': 81,
        'limitations': [
            'source_room_token is an original identifier only; no room geometry or physical identity is inferred.',
            'The T60 values are unchanged historical test labels. Filename rt tokens are not substituted as new labels.',
            'Cases are not paired speech/SNR controls. Both short and long simulated recommendations lack 0 dB samples.',
            'The long simulated recommendation has five 20 dB samples while real R61 has none; a future simulated/real error difference cannot be attributed to RIR type alone.',
            'Existing provenance and archive migration limitations remain; no audio identity or RIR hashes were recomputed.',
        ],
        'new_training_inference_synthesis_or_label_calculation': False,
        'input_and_manuscript_sha256': before, 'inputs_and_manuscripts_unchanged': True,
    }
    (OUT / 'selection_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'eligible': len(eligible), 'recommended': [{k:r[k] for k in ['candidate_case_id','historical_t60_s','original_filename','n_test2_test4','n_test2','n_test4','snr_counts_test2_test4']} for r in selected], 'recommended_sim_samples': len(sample_rows), 'model_metrics_computed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
