#!/usr/bin/env python3
"""Paper-grade layerwise CKA analysis for CMGAN-T60.

The script extracts representations for all utterances in test1--test4,
computes paired-bootstrap confidence intervals, per-split and pooled CKA,
and writes machine-readable tables plus publication-ready figures.
Feature files are cached so interrupted runs can resume safely.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import CMGANT60Dataset
from models.generator_t60 import (
    TSCNet_KAN_MultiTask_2TSCB,
    TSCNet_T60Estimator_2TSCB,
)


DATASET_ROOT = "/mnt/tidal-sh01/usr/chuan/youling/project/dataset/T60_Dataset_v7_4s"
SPLITS = ("test1", "test2", "test3", "test4")
LAYERS = ("encoder", "tscb_1", "tscb_2")
LAYER_LABELS = {"encoder": "Dense Encoder", "tscb_1": "TSCB-1", "tscb_2": "TSCB-2"}
MODELS = {
    "single_task": {
        "checkpoint": ROOT / "runs/single_t60_kan_v2_mse/best_model.pth",
        "label": "Single-task",
        "kind": "single",
        "color": "#666666",
    },
    "norm_w_denoise": {
        "checkpoint": ROOT / "runs/kan_multitask_norm_w_denoise_b005/best_model.pth",
        "label": "Denoise-dominant (alpha=1, beta=0.05)",
        "kind": "multi",
        "color": "#1976D2",
    },
    "norm_balanced": {
        "checkpoint": ROOT / "runs/kan_multitask_norm_balanced/best_model.pth",
        "label": "Balanced (alpha=1, beta=1)",
        "kind": "multi",
        "color": "#388E3C",
    },
    "norm_w_t60": {
        "checkpoint": ROOT / "runs/kan_multitask_norm_w_t60/best_model.pth",
        "label": "T60-dominant (alpha=0.1, beta=1)",
        "kind": "multi",
        "color": "#D32F2F",
    },
}


def build_model(spec, device):
    kwargs = dict(
        t60_head_type="fourier_kan",
        t60_fourier_proj_dim=64,
        t60_fourier_hidden_dims=(32, 16),
        t60_fourier_first_num_frequencies=16,
        t60_fourier_hidden_num_frequencies=8,
        t60_fourier_dropout=0.1,
        t60_out_activation="sigmoid",
    )
    cls = TSCNet_T60Estimator_2TSCB if spec["kind"] == "single" else TSCNet_KAN_MultiTask_2TSCB
    model = cls(**kwargs)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"  state_dict warning: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}", flush=True)
    return model.eval().to(device)


def make_dataset(split):
    return CMGANT60Dataset(
        dataset_root=DATASET_ROOT,
        split=split,
        n_fft=400,
        hop_length=100,
        audio_length=4.0,
        target_sr=16000,
        t60_range=(0.1, 1.5),
    )


def extract_features(model, loader, device):
    features = {layer: [] for layer in LAYERS}
    targets = []
    activations = {}

    def hook_for(name):
        def hook(_module, _inputs, output):
            features_tensor = output.detach().float().mean(dim=(-1, -2)).cpu()
            activations[name] = features_tensor
        return hook

    hooks = [
        model.dense_encoder.register_forward_hook(hook_for("encoder")),
        model.TSCB_1.register_forward_hook(hook_for("tscb_1")),
        model.TSCB_2.register_forward_hook(hook_for("tscb_2")),
    ]
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, 1):
            _ = model(batch["noisy_spec"].to(device, non_blocking=True))
            for layer in LAYERS:
                features[layer].append(activations[layer].numpy())
            targets.append(batch["t60_raw"].numpy())
            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"    batches {batch_idx}/{len(loader)}", flush=True)
    for hook in hooks:
        hook.remove()
    result = {layer: np.concatenate(features[layer]).astype(np.float32) for layer in LAYERS}
    result["t60_raw"] = np.concatenate(targets).astype(np.float32)
    return result


def feature_space_linear_cka(x, y):
    """Centered linear CKA without materializing N x N Gram matrices."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = np.square(xy).sum()
    denominator = np.sqrt(np.square(xx).sum() * np.square(yy).sum())
    return float(numerator / max(denominator, np.finfo(np.float64).tiny))


def paired_bootstrap_cka(x, y, repeats, rng):
    n = len(x)
    values = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        indices = rng.integers(0, n, size=n)
        values[i] = feature_space_linear_cka(x[indices], y[indices])
    return values


def load_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def extract_or_load_all(args, features_dir, device):
    all_features = {split: {} for split in SPLITS}
    for split in SPLITS:
        dataset = make_dataset(split)
        print(f"\n[{split}] {len(dataset)} utterances", flush=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for model_key in args.models:
            cache = features_dir / f"{split}__{model_key}.npz"
            if cache.exists() and not args.force_extract:
                print(f"  Loading cached features: {cache.name}", flush=True)
                all_features[split][model_key] = load_npz(cache)
                continue
            print(f"  Extracting {model_key}", flush=True)
            model = build_model(MODELS[model_key], device)
            values = extract_features(model, loader, device)
            np.savez_compressed(cache, **values)
            all_features[split][model_key] = values
            del model
            torch.cuda.empty_cache()
    return all_features


def compute_results(all_features, model_keys, bootstrap_repeats, seed):
    multi_models = [key for key in model_keys if key != "single_task"]
    rows = []
    bootstrap_store = {}
    for split_idx, split in enumerate(SPLITS):
        reference = all_features[split]["single_task"]
        for model_idx, model_key in enumerate(multi_models):
            for layer_idx, layer in enumerate(LAYERS):
                x = reference[layer]
                y = all_features[split][model_key][layer]
                if x.shape[0] != y.shape[0]:
                    raise ValueError(f"Sample mismatch: {split}/{model_key}/{layer}: {x.shape} vs {y.shape}")
                point = feature_space_linear_cka(x, y)
                cell_seed = seed + split_idx * 1000 + model_idx * 100 + layer_idx
                boot = paired_bootstrap_cka(x, y, bootstrap_repeats, np.random.default_rng(cell_seed))
                low, high = np.percentile(boot, [2.5, 97.5])
                key = f"{split}__{model_key}__{layer}"
                bootstrap_store[key] = boot.astype(np.float32)
                rows.append({
                    "scope": "split",
                    "split": split,
                    "model": model_key,
                    "layer": layer,
                    "n_samples": len(x),
                    "cka": point,
                    "bootstrap_mean": float(boot.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_repeats": bootstrap_repeats,
                })
                print(f"  {split} {model_key} {layer}: {point:.4f} [{low:.4f}, {high:.4f}]", flush=True)

    # Macro summaries treat the four test sets as four distinct conditions.
    for model_key in multi_models:
        for layer in LAYERS:
            values = [r["cka"] for r in rows if r["scope"] == "split" and r["model"] == model_key and r["layer"] == layer]
            rows.append({
                "scope": "macro",
                "split": "test1-test4",
                "model": model_key,
                "layer": layer,
                "n_samples": 4320,
                "cka": float(np.mean(values)),
                "bootstrap_mean": "",
                "ci95_low": "",
                "ci95_high": "",
                "bootstrap_repeats": "",
                "across_split_sd": float(np.std(values, ddof=1)),
            })

    # Pooled CKA is an explicitly labeled sensitivity check, not the primary result.
    for model_key in multi_models:
        for layer in LAYERS:
            x = np.concatenate([all_features[s]["single_task"][layer] for s in SPLITS])
            y = np.concatenate([all_features[s][model_key][layer] for s in SPLITS])
            rows.append({
                "scope": "pooled",
                "split": "test1+test2+test3+test4",
                "model": model_key,
                "layer": layer,
                "n_samples": len(x),
                "cka": feature_space_linear_cka(x, y),
                "bootstrap_mean": "",
                "ci95_low": "",
                "ci95_high": "",
                "bootstrap_repeats": "",
            })
    return rows, bootstrap_store


def write_csv(rows, path):
    fields = [
        "scope", "split", "model", "layer", "n_samples", "cka",
        "bootstrap_mean", "ci95_low", "ci95_high", "bootstrap_repeats",
        "across_split_sd",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_primary(rows, out_dir):
    model_key = "norm_w_denoise"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    offsets = np.linspace(-0.12, 0.12, len(SPLITS))
    colors = ["#1565C0", "#00897B", "#EF6C00", "#8E24AA"]
    x_base = np.arange(len(LAYERS))
    for split, offset, color in zip(SPLITS, offsets, colors):
        selected = [next(r for r in rows if r["scope"] == "split" and r["split"] == split and r["model"] == model_key and r["layer"] == layer) for layer in LAYERS]
        values = np.array([r["cka"] for r in selected])
        lower = values - np.array([r["ci95_low"] for r in selected])
        upper = np.array([r["ci95_high"] for r in selected]) - values
        ax.errorbar(x_base + offset, values, yerr=np.vstack([lower, upper]), marker="o", capsize=3, linewidth=1.6, label=split, color=color)
    ax.set_xticks(x_base, [LAYER_LABELS[layer] for layer in LAYERS])
    ax.set_ylabel("Centered linear CKA")
    ax.set_ylim(0, 1.02)
    ax.set_title("Single-task vs. denoise-dominant multi-task representations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_primary_b005_per_split_ci95.png", dpi=300)
    fig.savefig(out_dir / "fig_primary_b005_per_split_ci95.pdf")
    plt.close(fig)


def plot_all_models_macro(rows, out_dir):
    models = [key for key in MODELS if key != "single_task"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(LAYERS))
    for model_key in models:
        selected = [next(r for r in rows if r["scope"] == "macro" and r["model"] == model_key and r["layer"] == layer) for layer in LAYERS]
        values = [r["cka"] for r in selected]
        errors = [r["across_split_sd"] for r in selected]
        ax.errorbar(x, values, yerr=errors, marker="o", capsize=3, linewidth=1.8, color=MODELS[model_key]["color"], label=MODELS[model_key]["label"])
    ax.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYERS])
    ax.set_ylabel("Macro-mean centered linear CKA")
    ax.set_ylim(0, 1.02)
    ax.set_title("Representation similarity across weighting configurations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_all_models_macro_mean_sd.png", dpi=300)
    fig.savefig(out_dir / "fig_all_models_macro_mean_sd.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    if "single_task" not in args.models:
        parser.error("--models must include single_task as the CKA reference")
    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f"unknown model keys: {sorted(unknown)}")
    missing = [str(MODELS[key]["checkpoint"]) for key in args.models if not MODELS[key]["checkpoint"].exists()]
    if missing:
        parser.error(f"missing checkpoints: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    features_dir = args.output / "features"
    figures_dir = args.output / "figures"
    features_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; bootstrap={args.bootstrap}; seed={args.seed}", flush=True)

    all_features = extract_or_load_all(args, features_dir, device)
    print("\nComputing CKA and paired bootstrap intervals", flush=True)
    rows, bootstrap_store = compute_results(all_features, args.models, args.bootstrap, args.seed)
    write_csv(rows, args.output / "cka_results.csv")
    np.savez_compressed(args.output / "bootstrap_distributions.npz", **bootstrap_store)
    metadata = {
        "dataset_root": DATASET_ROOT,
        "splits": list(SPLITS),
        "samples_per_split": {split: len(make_dataset(split)) for split in SPLITS},
        "models": {key: {k: str(v) for k, v in spec.items() if k != "color"} for key, spec in MODELS.items()},
        "layers": list(LAYERS),
        "pooling": "global mean over time and frequency",
        "cka": "centered linear CKA in feature space",
        "bootstrap": {"repeats": args.bootstrap, "method": "paired resampling with replacement", "ci": "percentile 95%", "seed": args.seed},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_primary(rows, figures_dir)
    plot_all_models_macro(rows, figures_dir)
    print(f"\nDone: {args.output}", flush=True)


if __name__ == "__main__":
    main()
