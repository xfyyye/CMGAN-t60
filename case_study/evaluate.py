"""Explicit, offline evaluation of the four frozen Case Study models.

This file is an entry point only. Importing it never loads weights or evaluates.
Predictions use historical file-level reference labels, not verified channel truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case_study.common import (
    REPO_ROOT, config_sha256, data_config_sha256, data_root, git_revision, load_config,
    read_manifest, reserve_output, resolve_repo_path, sha256_file,
    validate_manifest, write_json,
)


# Explicit assets: no search for "latest" weights, training, or model selection.
FROZEN_ASSETS = {
    "configs/t60_multitask/kan_mse_norm_w_denoise_b005_seed42.yaml":
        ("multitask", "runs/kan_multitask_norm_w_denoise_b005_seed42/best_model.pth"),
    "configs/t60_multitask/mlp_matched_drop01_mse_norm_w_denoise_b005_seed42.yaml":
        ("multitask", "runs/mlp_matched_drop01_multitask_norm_w_denoise_b005_seed42/best_model.pth"),
    "configs/t60_single/kan_v2_mse_seed42.yaml":
        ("single", "runs/single_t60_kan_v2_mse_seed42/best_model.pth"),
    "configs/t60_single/mlp_matched_drop01_mse_seed42.yaml":
        ("single", "runs/single_t60_mlp_matched_drop01_mse_seed42/best_model.pth"),
}


def metrics(records):
    errors = [(r["prediction_t60_s"] - r["t60_s"]) * 1000 for r in records]
    if not errors or not all(math.isfinite(e) for e in errors):
        raise ValueError("Empty/nonfinite metric group")
    n = len(errors)
    return {"count": n, "rmse_ms": math.sqrt(math.fsum(e * e for e in errors) / n),
            "mae_ms": math.fsum(abs(e) for e in errors) / n,
            "bias_ms": math.fsum(errors) / n}


def grouped_metrics(records):
    result = {"overall": metrics(records)}
    for name, fields in (
        ("by_room", ("case_id", "room_name")),
        ("by_room_noise", ("case_id", "room_name", "noise_id")),
        ("by_room_snr", ("case_id", "room_name", "snr_db")),
        ("by_room_noise_snr", ("case_id", "room_name", "noise_id", "snr_db")),
    ):
        groups = defaultdict(list)
        for row in records:
            groups[tuple(row[f] for f in fields)].append(row)
        result[name] = [dict(zip(fields, key), **metrics(groups[key]))
                        for key in sorted(groups)]
    return result


def run(config_path):
    # Third-party numerical imports are intentionally deferred until execution.
    import numpy as np
    import torch
    import torchaudio
    import yaml
    from dataset import create_test_loader, T60Normalizer
    from models.generator_t60 import TSCNet_T60Estimator_2TSCB, TSCNet_KAN_MultiTask_2TSCB
    from utils import power_compress

    cfg = load_config(config_path)
    cfg_hash = config_sha256(cfg)
    data_cfg_hash = data_config_sha256(cfg)
    root = data_root(cfg).resolve()
    summary_path = root / "synthesis_summary.json"
    synthesis = json.loads(summary_path.read_text(encoding="utf-8"))
    if (synthesis.get("status") != "complete" or synthesis.get("sample_count") != 3240
            or synthesis.get("data_config_sha256") != data_cfg_hash):
        raise ValueError("Synthesis must be complete, contain 3240 samples, and match the data protocol")
    manifest_path = root / "manifest.jsonl"
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != synthesis.get("manifest_sha256"):
        raise ValueError("Manifest hash differs from the completed synthesis record")
    rows = read_manifest(manifest_path)
    validate_manifest(cfg, rows)
    if len(rows) != 3240:
        raise ValueError("Expected exactly 3240 manifest rows")
    synthesis_hash = sha256_file(summary_path)
    options = cfg["evaluation"]
    if options.get("precision", "float32") != "float32":
        raise ValueError("Only explicit float32 evaluation is supported")
    batch_size = int(options.get("batch_size", 8))
    workers = int(options.get("num_workers", 0))
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid batch_size/num_workers")
    verify_hashes = options.get("verify_audio_hashes", True)
    if not isinstance(verify_hashes, bool):
        raise ValueError("verify_audio_hashes must be boolean")
    device_name = options.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA is supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.get_device_properties(device)  # Validate device index before output creation.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)

    models = cfg["models"]
    expected = {resolve_repo_path(k).resolve(): (task, resolve_repo_path(cp).resolve())
                for k, (task, cp) in FROZEN_ASSETS.items()}
    if len(models) != 4 or len({m["id"] for m in models}) != 4:
        raise ValueError("Exactly four distinct frozen models are required")
    if {resolve_repo_path(m["config"]).resolve() for m in models} != set(expected):
        raise ValueError("Model configurations differ from the four frozen assets")
    assets = []
    for model_entry in models:
        ident = model_entry["id"]
        if not isinstance(ident, str) or not ident or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in ident):
            raise ValueError("Model id must be a safe filename component")
        config_file = resolve_repo_path(model_entry["config"]).resolve()
        checkpoint = resolve_repo_path(model_entry["checkpoint"]).resolve()
        if (model_entry["task"], checkpoint) != expected[config_file]:
            raise ValueError("Frozen checkpoint/task mapping mismatch")
        model_cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        mc = model_cfg["model"]
        expected_head = "mlp" if "mlp_matched" in config_file.name else "fourier_kan"
        if mc.get("t60_head_type") != expected_head or mc.get("t60_out_activation") != "sigmoid":
            raise ValueError("Frozen head type/output activation mismatch")
        if expected_head == "mlp":
            if (mc.get("t60_mlp_hidden_dims") != [192, 192, 64]
                    or mc.get("t60_dropout") != 0.1):
                raise ValueError("Matched MLP requires [192,192,64] and dropout 0.1")
        else:
            fourier_settings = {"t60_fourier_proj_dim": 64,
                                "t60_fourier_hidden_dims": [32, 16],
                                "t60_fourier_first_num_frequencies": 16,
                                "t60_fourier_hidden_num_frequencies": 8,
                                "t60_fourier_dropout": 0.1}
            if any(mc.get(key) != value for key, value in fourier_settings.items()):
                raise ValueError("Frozen Fourier-KAN head configuration mismatch")
        dc = model_cfg["data"]
        for key, value in {"n_fft": 400, "hop_length": 100, "audio_length": 4.0,
                           "target_sr": 16000, "t60_min": 0.1, "t60_max": 1.5}.items():
            if dc.get(key) != value:
                raise ValueError(f"Unexpected formal preprocessing setting: {key}")
        if dc.get("log_scale", False) or model_cfg["model"].get("n_tscb") != 2:
            raise ValueError("Expected linear labels and two TSC blocks")
        assets.append((model_entry, model_cfg, {"config_path": str(config_file),
                       "checkpoint_path": str(checkpoint), "config_sha256": sha256_file(config_file),
                       "checkpoint_sha256": sha256_file(checkpoint)}))

    cases = sorted({r["case_id"] for r in rows})
    per_case = {case: sorted((r for r in rows if r["case_id"] == case),
                            key=lambda r: r["sample_id"]) for case in cases}
    expected_order = [r["sample_id"] for case in cases for r in per_case[case]]

    def check_audio(case_rows):
        for row in case_rows:
            path = (root / row["noisy_relative_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Invalid audio path for {row['sample_id']}")
            if verify_hashes and sha256_file(path) != row["noisy_sha256"]:
                raise ValueError(f"Audio hash mismatch: {row['sample_id']}")

    check_audio(rows)
    output = resolve_repo_path(cfg["paths"]["output_root"]) / "evaluation"
    reserve_output(output)
    write_json(output / "config.snapshot.json", cfg)
    source_paths = [REPO_ROOT / name for name in (
        "case_study/evaluate.py", "case_study/common.py", "dataset.py", "utils.py",
        "test.py", "test_multitask.py",
    )]
    source_paths += sorted((REPO_ROOT / "models").rglob("*.py"))
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path)
                     for path in source_paths}
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "torchaudio": str(torchaudio.__version__), "numpy": str(np.__version__),
                   "device": str(device), "cuda_runtime": torch.version.cuda,
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                   "precision": "float32", "autocast": False, "tf32": False,
                   "batch_size": batch_size, "num_workers": workers,
                   "git_revision": git_revision(), "source_sha256": source_hashes}
    results = {}
    for entry, model_cfg, provenance in assets:
        for kind in ("config", "checkpoint"):
            if sha256_file(Path(provenance[kind + "_path"])) != provenance[kind + "_sha256"]:
                raise ValueError(f"Model {kind} changed during evaluation")
        mc = model_cfg["model"]
        kwargs = {"num_channel": 64, "num_features": 201,
                  "t60_head_type": mc["t60_head_type"],
                  "t60_hidden_dim": mc.get("t60_hidden_dim", 128),
                  "t60_mlp_hidden_dims": mc.get("t60_mlp_hidden_dims"),
                  "t60_dropout": mc.get("t60_dropout", 0.3),
                  "t60_fourier_proj_dim": mc.get("t60_fourier_proj_dim", 64),
                  "t60_fourier_hidden_dims": tuple(mc.get("t60_fourier_hidden_dims", (32, 16))),
                  "t60_fourier_first_num_frequencies": mc.get("t60_fourier_first_num_frequencies", 16),
                  "t60_fourier_hidden_num_frequencies": mc.get("t60_fourier_hidden_num_frequencies", 8),
                  "t60_fourier_dropout": mc.get("t60_fourier_dropout", 0.1),
                  "t60_out_activation": mc.get("t60_out_activation", "sigmoid")}
        constructor = TSCNet_T60Estimator_2TSCB if entry["task"] == "single" else TSCNet_KAN_MultiTask_2TSCB
        model = constructor(**kwargs)
        # Fail closed if a legacy checkpoint needs unsupported pickle globals.
        checkpoint = torch.load(provenance["checkpoint_path"], map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)
        del state, checkpoint
        model = model.to(device=device, dtype=torch.float32).eval()
        normalizer = T60Normalizer(0.1, 1.5, log_scale=False)
        records = []
        for case in cases:
            case_rows = per_case[case]
            check_audio(case_rows)
            loader = create_test_loader(test_split=case, dataset_root=str(root / "audio"),
                                        n_fft=400, hop_length=100, audio_length=4.0,
                                        target_sr=16000, t60_range=(0.1, 1.5), log_scale=False,
                                        batch_size=batch_size, num_workers=workers)
            names = [r["sample_id"] for r in case_rows]
            if list(loader.dataset.samples) != names:
                raise ValueError(f"Dataset sample set/order differs from manifest: {case}")
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    batch_names = list(batch["sample_name"])
                    batch_rows = case_rows[offset:offset + len(batch_names)]
                    if batch_names != [r["sample_id"] for r in batch_rows]:
                        raise ValueError("Batch sample order mismatch")
                    for i, row in enumerate(batch_rows):
                        if (abs(float(batch["t60_raw"][i]) - row["t60_s"]) > 1e-6
                                or abs(float(batch["snr"][i]) - row["snr_db"]) > 1e-6):
                            raise ValueError("Filename-derived T60/SNR differs from manifest")
                    spec = batch["noisy_spec"].to(device=device, dtype=torch.float32)
                    inputs = power_compress(spec.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)
                    prediction = model(inputs)
                    if entry["task"] == "multitask":
                        prediction = prediction[2]
                    values = normalizer.denormalize(prediction.detach().cpu().numpy()).reshape(-1)
                    if len(values) != len(batch_rows) or not np.isfinite(values).all():
                        raise ValueError("Invalid prediction shape or nonfinite prediction")
                    for row, value in zip(batch_rows, values):
                        records.append({**row, "model_id": entry["id"],
                                        "prediction_t60_s": float(value),
                                        "error_ms": (float(value) - row["t60_s"]) * 1000,
                                        "reference_label_status": "historical_file_reference_not_verified_channel_ground_truth",
                                        "model_config_sha256": provenance["config_sha256"],
                                        "checkpoint_sha256": provenance["checkpoint_sha256"]})
                    offset += len(batch_names)
            if offset != len(case_rows):
                raise ValueError("Incomplete case evaluation")
        if [r["sample_id"] for r in records] != expected_order:
            raise ValueError("Model predictions do not cover the identical manifest order")
        check_audio(rows)
        prediction_path = output / f"{entry['id']}.predictions.jsonl"
        with prediction_path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        result = {"model": entry, "model_configuration": model_cfg,
                  **provenance, "environment": environment,
                  "evaluation_config_sha256": cfg_hash, "data_config_sha256": data_cfg_hash,
                  "sample_count": len(records), "metrics": grouped_metrics(records),
                  "prediction_file": prediction_path.name, "prediction_sha256": sha256_file(prediction_path)}
        write_json(output / f"{entry['id']}.summary.json", result)
        results[entry["id"]] = result
        del model

    if sha256_file(manifest_path) != manifest_hash or sha256_file(summary_path) != synthesis_hash:
        raise ValueError("Synthesis provenance changed during evaluation")
    for relative, expected_hash in source_hashes.items():
        if sha256_file(REPO_ROOT / relative) != expected_hash:
            raise ValueError(f"Evaluation source changed during execution: {relative}")
    for _, _, provenance in assets:
        for kind in ("config", "checkpoint"):
            if sha256_file(Path(provenance[kind + "_path"])) != provenance[kind + "_sha256"]:
                raise ValueError(f"Model {kind} changed during evaluation")
    write_json(output / "summary.json", {
        "status": "complete", "sample_count": len(rows), "prediction_count": len(rows) * len(results),
        "config_sha256": cfg_hash, "data_config_sha256": data_cfg_hash,
        "synthesis_config_sha256": synthesis.get("config_sha256"),
        "manifest_sha256": manifest_hash,
        "synthesis_summary_sha256": synthesis_hash, "verify_audio_hashes": verify_hashes,
        "label_status": "historical_file_reference_not_verified_channel_ground_truth",
        "environment": environment, "models": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Case Study configuration path")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
