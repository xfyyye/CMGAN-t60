"""Standard-library-only configuration, provenance and manifest validation."""
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'case_study' / 'config.json'
OUTPUT_BASE = REPO_ROOT / 'case_study' / 'outputs'
LABEL_POLICY = 'unknown_file_level_historical'
MODEL_SPECS = {
    'dact_kan_multitask': ('multitask', 'configs/t60_multitask/kan_mse_norm_w_denoise_b005_seed42.yaml',
                           'runs/kan_multitask_norm_w_denoise_b005_seed42/best_model.pth'),
    'matched_mlp_multitask': ('multitask', 'configs/t60_multitask/mlp_matched_drop01_mse_norm_w_denoise_b005_seed42.yaml',
                            'runs/mlp_matched_drop01_multitask_norm_w_denoise_b005_seed42/best_model.pth'),
    'kan_single': ('single', 'configs/t60_single/kan_v2_mse_seed42.yaml',
                   'runs/single_t60_kan_v2_mse_seed42/best_model.pth'),
    'matched_mlp_single': ('single', 'configs/t60_single/mlp_matched_drop01_mse_seed42.yaml',
                          'runs/single_t60_mlp_matched_drop01_mse_seed42/best_model.pth'),
}
CASE_SPECS = {
    'office': (0.454, 'T60_0.454_000.wav', 'air_type1_air_binaural_office_0_1.wav', 78),
    'lecture': (0.773, 'T60_0.773.wav', 'air_type1_air_binaural_lecture_1_1.wav', 109),
    'stairway': (1.138, 'T60_1.138.wav', 'air_type1_air_binaural_stairway_1_3_45.wav', 123),
}


def resolve_repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def raw_path(cfg, relative):
    root = resolve_repo_path(cfg['paths']['raw_root'])
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError(f'Expected a path relative to raw_root: {relative}')
    result = (root / value).resolve()
    if not _within(result, root):
        raise ValueError(f'Raw path escapes raw_root: {relative}')
    return result


def data_root(cfg):
    return resolve_repo_path(cfg['paths']['output_root']) / 'data'


def _finite(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected a finite number for {context}: {value!r}')
    return value


def load_config(path=DEFAULT_CONFIG):
    with resolve_repo_path(path).open(encoding='utf-8') as stream:
        cfg = json.load(stream)
    if cfg.get('schema_version') != 1 or cfg.get('experiment_id') != 'air_three_rooms_ch0_v1':
        raise ValueError('Unsupported case-study schema or experiment_id')
    output = resolve_repo_path(cfg['paths']['output_root'])
    base = OUTPUT_BASE.resolve()
    if output == base or not _within(output, base):
        raise ValueError('output_root must be a versioned directory under case_study/outputs')
    raw = resolve_repo_path(cfg['paths']['raw_root'])
    if _within(output, raw) or _within(raw, output):
        raise ValueError('Input raw_root and output_root must not overlap')
    synth = cfg['synthesis']
    fixed = {'sample_rate_hz': 16000, 'seed': 42, 'channel_index': 0,
             'speech_count': 90, 'speech_input_samples': 128000,
             'crop_start_sample': 32000, 'output_samples': 64000,
             'expected_samples': 3240, 'wav_subtype': 'FLOAT'}
    for key, value in fixed.items():
        if synth.get(key) != value or isinstance(synth.get(key), bool):
            raise ValueError(f'This protocol requires synthesis.{key}={value!r}')
    if synth['snr_db'] != [-5, 0, 5, 10, 15, 20]:
        raise ValueError('The approved protocol requires all six SNR levels')
    if not 0 < _finite(synth['peak_limit'], 'peak_limit') < 1:
        raise ValueError('peak_limit must lie strictly between zero and one')
    for key in ['min_power', 'snr_tolerance_db', 'additivity_tolerance']:
        if _finite(synth[key], key) <= 0:
            raise ValueError(f'{key} must be positive')
    raw_path(cfg, synth['speech_directory'])
    noises = synth['noise_files']
    if len(noises) != 2 or [n['id'] for n in noises] != ['white', 'babble']:
        raise ValueError('The approved noise order is white, babble')
    for item in noises:
        expected = f"Noise/noise92_10_test_noise_unseen/{item['id']}.wav"
        if item['path'] != expected:
            raise ValueError(f'Noise source differs from approved protocol: {item}')
        raw_path(cfg, item['path'])
    cases = cfg['cases']
    if len(cases) != 3 or {c['id'] for c in cases} != set(CASE_SPECS):
        raise ValueError('Exactly office, lecture and stairway are required')
    for case in cases:
        t60, filename, original, line = CASE_SPECS[case['id']]
        if (case['t60_s'], Path(case['rir_path']).name, case['original_filename'],
                case['label_csv_line'], case['expected_channels']) != (t60, filename, original, line, 2):
            raise ValueError(f'RIR or label metadata differs from approved case: {case["id"]}')
        raw_path(cfg, case['rir_path'])
        raw_path(cfg, case['label_csv'])
    models = cfg['models']
    if len(models) != 4 or {m['id'] for m in models} != set(MODEL_SPECS):
        raise ValueError('Exactly the four approved internal models are required')
    for model in models:
        if (model['task'], model['config'], model['checkpoint']) != MODEL_SPECS[model['id']]:
            raise ValueError(f'Model/config/checkpoint mapping differs: {model["id"]}')
    evaluation = cfg['evaluation']
    if evaluation['precision'] != 'float32' or evaluation['verify_audio_hashes'] is not True:
        raise ValueError('Evaluation requires float32 and audio hash verification')
    if (type(evaluation['batch_size']) is not int or evaluation['batch_size'] < 1
            or type(evaluation['num_workers']) is not int or evaluation['num_workers'] < 0):
        raise ValueError('Invalid evaluation batch_size or num_workers')
    return cfg


def config_sha256(cfg):
    encoded = json.dumps(cfg, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def data_config_sha256(cfg):
    """Bind generated audio to its protocol, independent of host/batch/device."""
    contract = {key: cfg[key] for key in ['schema_version', 'experiment_id', 'synthesis', 'cases']}
    return config_sha256(contract)


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    path = Path(path)
    content = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)


def reserve_output(path):
    path = Path(path).resolve()
    base = OUTPUT_BASE.resolve()
    if path == base or not _within(path, base):
        raise ValueError('Output must stay below case_study/outputs')
    path.mkdir(parents=True, exist_ok=False)


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def read_manifest(path):
    result = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Empty manifest line {line_number}')
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Manifest line {line_number} is not an object')
            result.append(row)
    return result


def validate_manifest(cfg, rows):
    synth = cfg['synthesis']
    if len(rows) != synth['expected_samples']:
        raise ValueError(f'Expected 3240 manifest rows, got {len(rows)}')
    cases = {c['id']: c for c in cfg['cases']}
    noises = {n['id']: n for n in synth['noise_files']}
    root = data_root(cfg).resolve()
    ids, grid, speeches, templates, audio_paths = set(), set(), {}, {}, set()
    for row in rows:
        cid, sid, nid = row['case_id'], row['speech_id'], row['noise_id']
        if cid not in cases or nid not in noises or not isinstance(sid, str) or not sid:
            raise ValueError('Unknown case, noise or speech identifier')
        sample = row['sample_id']
        if not isinstance(sample, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', sample) or sample in ids:
            raise ValueError(f'Invalid or duplicate sample_id: {sample}')
        ids.add(sample)
        snr = _finite(row['snr_db'], 'manifest SNR')
        if snr not in synth['snr_db']:
            raise ValueError(f'Unapproved SNR: {snr}')
        t60 = _finite(row['t60_s'], 'manifest label')
        if abs(t60 - cases[cid]['t60_s']) > 1e-9 or row['channel_index'] != 0:
            raise ValueError(f'Case label or channel mismatch: {sample}')
        if row['label_channel_policy'] != LABEL_POLICY:
            raise ValueError('Historical label/channel uncertainty must remain explicit')
        parts = sample.split('_')
        if abs(float(parts[-2]) - t60) > 1e-9 or float(parts[-1].removesuffix('dB')) != snr:
            raise ValueError(f'Loader filename label/SNR disagrees with manifest: {sample}')
        key = (cid, sid, nid, snr)
        if key in grid:
            raise ValueError(f'Duplicate condition: {key}')
        grid.add(key)
        speech = (row['speech_relative_path'], row['speech_sha256'])
        if sid in speeches and speeches[sid] != speech:
            raise ValueError(f'Speech source changes across conditions: {sid}')
        speeches[sid] = speech
        if row['noise_source_relative_path'] != noises[nid]['path']:
            raise ValueError(f'Noise source mismatch: {sample}')
        offset = row['noise_offset_samples']
        if type(offset) is not int or offset < 0:
            raise ValueError(f'Invalid noise offset: {sample}')
        template = (row['noise_source_relative_path'], row['noise_source_sha256'], offset)
        base_pair = (sid, nid)
        if base_pair in templates and templates[base_pair] != template:
            raise ValueError(f'Noise window changes across rooms/SNRs: {base_pair}')
        templates[base_pair] = template
        for field in ['sample_rate_hz', 'output_samples', 'crop_start_sample']:
            if row[field] != synth[field]:
                raise ValueError(f'Observation format mismatch: {sample}/{field}')
        folder = Path('audio') / cid / sample
        expected = {'noisy_relative_path': folder / f'{sample}.wav',
                    'clean_relative_path': folder / f'{sample}_denoised.wav',
                    'noise_relative_path': folder / f'{sample}_noise.wav'}
        for field, value in expected.items():
            relative = Path(row[field])
            if relative != value or not _within((root / relative).resolve(), root):
                raise ValueError(f'Audio path mismatch or escape: {sample}/{field}')
            if str(relative) in audio_paths:
                raise ValueError(f'Duplicate output path: {relative}')
            audio_paths.add(str(relative))
        if not re.fullmatch(r'[0-9a-f]{64}', row['noisy_sha256']):
            raise ValueError(f'Invalid noisy SHA256: {sample}')
    if len(speeches) != 90 or len(set(speeches.values())) != 90:
        raise ValueError('Exactly 90 unique speech files are required')
    expected_grid = set(product(cases, speeches, noises, synth['snr_db']))
    if grid != expected_grid or len(templates) != 180:
        raise ValueError('Manifest is not the complete paired 3 x 90 x 2 x 6 design')
    if Counter(r['case_id'] for r in rows) != Counter({cid: 1080 for cid in cases}):
        raise ValueError('Each case must have exactly 1080 samples')
