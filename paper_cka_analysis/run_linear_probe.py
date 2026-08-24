#!/usr/bin/env python3
"""Paper-grade linear probing for CMGAN-T60 (companion to run_paper_cka.py).

Purpose: quantify representation-level benefit of the multi-task model.
A ridge probe is fitted on FROZEN layer features to regress T60, and the
held-out R^2 of the single-task baseline is compared with the multi-task
models. Two pooling modes are probed:

  mean      : global mean over (T, F)      -> C dims
  multistats: mean over F, then avg/max/std over T -> 3C dims
              (identical to how the real T60 head reads the bottleneck)

Protocols:
  pooled : random 80/20 split over the 4320 pooled utterances,
           repeated with n_seeds seeds, reported as mean +/- SD
  l1o    : leave-one-split-out (train on three test sets, evaluate on the
           held-out one), deterministic, measures cross-condition decodability

Features are cached so interrupted runs resume safely.
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

from run_paper_cka import (
    LAYERS,
    LAYER_LABELS,
    MODELS,
    SPLITS,
    build_model,
    make_dataset,
)

POOLINGS = ("mean", "multistats")


def extract_probe_features(model, loader, device):
    """Per-utterance mean and avg/max/std statistics of each layer."""
    feats = {p: {layer: [] for layer in LAYERS} for p in POOLINGS}
    targets = []
    pocket = {}

    def hook_for(name):
        def hook(_module, _inputs, output):
            x = output.detach().float()                     # (B, C, T, F)
            mean_tf = x.mean(dim=(-1, -2))                  # (B, C)
            mean_f = x.mean(dim=-1)                         # (B, C, T)
            avg_t = mean_f.mean(dim=-1)
            max_t = mean_f.max(dim=-1).values
            std_t = mean_f.std(dim=-1)
            pocket[name] = {
                "mean": mean_tf.cpu().numpy(),
                "multistats": torch.cat([avg_t, max_t, std_t], dim=1).cpu().numpy(),
            }
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
                for p in POOLINGS:
                    feats[p][layer].append(pocket[layer][p])
            targets.append(batch["t60_raw"].numpy())
            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"    batches {batch_idx}/{len(loader)}", flush=True)
    for hook in hooks:
        hook.remove()
    result = {p: {layer: np.concatenate(feats[p][layer]).astype(np.float32)
                  for layer in LAYERS} for p in POOLINGS}
    result["t60_raw"] = np.concatenate(targets).astype(np.float32)
    return result


def ridge_r2(x_tr, y_tr, x_te, y_te, alpha):
    """Closed-form ridge on standardised features; returns held-out R^2."""
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0) + 1e-8
    x_tr_s = (x_tr - mu) / sd
    x_te_s = (x_te - mu) / sd
    y_mu = y_tr.mean()
    gram = x_tr_s.T @ x_tr_s + alpha * np.eye(x_tr_s.shape[1], dtype=np.float64)
    w = np.linalg.solve(gram, x_tr_s.T.astype(np.float64) @ (y_tr - y_mu))
    pred = x_te_s.astype(np.float64) @ w + y_mu
    ss_res = float(np.square(y_te - pred).sum())
    ss_tot = float(np.square(y_te - y_te.mean()).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def load_npz(path):
    """Load flattened caches and the nested object caches from the first version."""
    with np.load(path, allow_pickle=True) as data:
        if all(f"{pooling}__{layer}" in data.files
               for pooling in POOLINGS for layer in LAYERS):
            result = {
                pooling: {layer: data[f"{pooling}__{layer}"] for layer in LAYERS}
                for pooling in POOLINGS
            }
            result["t60_raw"] = data["t60_raw"]
            return result
        result = {pooling: data[pooling].item() for pooling in POOLINGS}
        result["t60_raw"] = data["t60_raw"]
        return result


def save_npz(path, values):
    """Store numeric arrays only; avoid pickled nested dictionaries."""
    flat = {
        f"{pooling}__{layer}": values[pooling][layer]
        for pooling in POOLINGS for layer in LAYERS
    }
    flat["t60_raw"] = values["t60_raw"]
    np.savez_compressed(path, **flat)


def extract_or_load_all(args, cache_dir, device):
    all_feats = {split: {} for split in SPLITS}
    for split in SPLITS:
        dataset = make_dataset(split)
        print(f"\n[{split}] {len(dataset)} utterances", flush=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        for model_key in args.models:
            cache = cache_dir / f"{split}__{model_key}.npz"
            if cache.exists() and not args.force_extract:
                print(f"  Loading cached probe features: {cache.name}", flush=True)
                all_feats[split][model_key] = load_npz(cache)
                continue
            print(f"  Extracting {model_key}", flush=True)
            model = build_model(MODELS[model_key], device)
            values = extract_probe_features(model, loader, device)
            save_npz(cache, values)
            all_feats[split][model_key] = values
            del model
            torch.cuda.empty_cache()
    return all_feats


def pooled_probe(all_feats, model_keys, pooling, n_seeds, base_seed, alpha, train_frac):
    rows = []
    for layer in LAYERS:
        for model_key in model_keys:
            x = np.concatenate([all_feats[s][model_key][pooling][layer] for s in SPLITS])
            y = np.concatenate([all_feats[s][model_key]["t60_raw"] for s in SPLITS])
            values = []
            for seed in range(base_seed, base_seed + n_seeds):
                rng = np.random.default_rng(seed)
                idx = rng.permutation(len(x))
                n_tr = int(round(train_frac * len(x)))
                tr, te = idx[:n_tr], idx[n_tr:]
                values.append(ridge_r2(x[tr], y[tr], x[te], y[te], alpha))
            values = np.asarray(values)
            rows.append({
                "protocol": "pooled",
                "pooling": pooling,
                "layer": layer,
                "model": model_key,
                "n_seeds": n_seeds,
                "r2_mean": float(values.mean()),
                "r2_sd": float(values.std(ddof=1)) if n_seeds > 1 else 0.0,
                "r2_per_seed": ";".join(f"{v:.4f}" for v in values),
            })
            print(f"  pooled/{pooling}/{layer}/{model_key}: "
                  f"R2 = {values.mean():.4f} +/- {rows[-1]['r2_sd']:.4f}", flush=True)
    return rows


def l1o_probe(all_feats, model_keys, pooling, alpha):
    rows = []
    for layer in LAYERS:
        for model_key in model_keys:
            values = []
            for held in SPLITS:
                tr_s = [s for s in SPLITS if s != held]
                x_tr = np.concatenate([all_feats[s][model_key][pooling][layer] for s in tr_s])
                y_tr = np.concatenate([all_feats[s][model_key]["t60_raw"] for s in tr_s])
                x_te = all_feats[held][model_key][pooling][layer]
                y_te = all_feats[held][model_key]["t60_raw"]
                values.append(ridge_r2(x_tr, y_tr, x_te, y_te, alpha))
            values = np.asarray(values)
            rows.append({
                "protocol": "l1o",
                "pooling": pooling,
                "layer": layer,
                "model": model_key,
                "n_seeds": 0,
                "r2_mean": float(values.mean()),
                "r2_sd": float(values.std(ddof=1)),
                "r2_per_seed": ";".join(f"{s}={v:.4f}" for s, v in zip(SPLITS, values)),
            })
            print(f"  l1o/{pooling}/{layer}/{model_key}: "
                  f"R2 = {values.mean():.4f} +/- {values.std(ddof=1):.4f}", flush=True)
    return rows


def paired_deltas(rows, model_keys, primary="norm_w_denoise"):
    deltas = []
    lookup = {(r["protocol"], r["pooling"], r["layer"], r["model"]): r for r in rows}
    for protocol in ("pooled", "l1o"):
        for pooling in POOLINGS:
            for layer in LAYERS:
                a = lookup.get((protocol, pooling, layer, primary))
                b = lookup.get((protocol, pooling, layer, "single_task"))
                if a is None or b is None:
                    continue
                deltas.append({
                    "protocol": protocol, "pooling": pooling, "layer": layer,
                    "delta_mean": a["r2_mean"] - b["r2_mean"],
                    "delta_sd": float(np.hypot(a["r2_sd"], b["r2_sd"])),
                })
                print(f"  delta[{protocol}/{pooling}/{layer}] "
                      f"{primary} - single_task = "
                      f"{deltas[-1]['delta_mean']:+.4f} +/- {deltas[-1]['delta_sd']:.4f}",
                      flush=True)
    return deltas


def plot_probes(rows, out_dir):
    models = [k for k in MODELS if k != "single_task"]
    fig, axes = plt.subplots(1, len(POOLINGS), figsize=(12.5, 4.6), sharey=True)
    for ax, pooling in zip(axes, POOLINGS):
        x = np.arange(len(LAYERS))
        width = 0.8 / (len(models) + 1)
        sel = [r for r in rows
               if r["protocol"] == "pooled" and r["pooling"] == pooling]
        for i, key in enumerate(["single_task"] + models):
            vals = [next(r for r in sel if r["layer"] == layer and r["model"] == key)
                    for layer in LAYERS]
            means = np.array([v["r2_mean"] for v in vals])
            sds = np.array([v["r2_sd"] for v in vals])
            color = MODELS[key]["color"] if key in MODELS else "#666666"
            ax.bar(x + i * width, means, width, yerr=sds, capsize=2,
                   color=color, alpha=0.85,
                   label=MODELS[key]["label"] if key in MODELS else "Single-task")
        ax.set_xticks(x + 0.4 - width / 2, [LAYER_LABELS[l] for l in LAYERS])
        ax.set_title(f"{pooling}-pooled probe")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Held-out R^2 (T60 linear probe)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_probe_r2_pooled.png", dpi=300)
    fig.savefig(out_dir / "fig_probe_r2_pooled.pdf")
    plt.close(fig)


def write_csv(rows, deltas, path, delta_path):
    fields = ["protocol", "pooling", "layer", "model", "n_seeds",
              "r2_mean", "r2_sd", "r2_per_seed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with delta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protocol", "pooling", "layer", "delta_mean", "delta_sd"])
        for d in deltas:
            writer.writerow([d["protocol"], d["pooling"], d["layer"],
                             f"{d['delta_mean']:.6f}", f"{d['delta_sd']:.6f}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    if "single_task" not in args.models:
        parser.error("--models must include single_task as the probe reference")
    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f"unknown model keys: {sorted(unknown)}")
    missing = [str(MODELS[k]["checkpoint"]) for k in args.models
               if not MODELS[k]["checkpoint"].exists()]
    if missing:
        parser.error(f"missing checkpoints: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "probe_features"
    figures_dir = args.output / "figures"
    cache_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seeds={args.n_seeds} (base {args.seed}); "
          f"ridge alpha={args.ridge_alpha}", flush=True)

    all_feats = extract_or_load_all(args, cache_dir, device)

    rows = []
    print("\nPooled 80/20 probe (mean over seeds)", flush=True)
    for pooling in POOLINGS:
        rows += pooled_probe(all_feats, args.models, pooling,
                             args.n_seeds, args.seed, args.ridge_alpha, args.train_frac)
    print("\nLeave-one-split-out probe (deterministic)", flush=True)
    for pooling in POOLINGS:
        rows += l1o_probe(all_feats, args.models, pooling, args.ridge_alpha)

    print("\nPaired deltas vs single_task", flush=True)
    deltas = paired_deltas(rows, args.models)

    write_csv(rows, deltas, args.output / "probe_results.csv",
              args.output / "probe_deltas.csv")
    plot_probes(rows, figures_dir)
    metadata = {
        "protocol_pooled": {"train_frac": args.train_frac, "n_seeds": args.n_seeds,
                            "base_seed": args.seed},
        "protocol_l1o": "train on three test sets, evaluate on held-out",
        "poolings": {
            "mean": "global mean over (T, F), C dims",
            "multistats": "mean over F then avg/max/std over T, 3C dims (matches T60 head)",
        },
        "regressor": f"ridge (alpha={args.ridge_alpha}) on standardised features",
        "models": {k: str(MODELS[k]["checkpoint"]) for k in args.models},
    }
    (args.output / "probe_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone: {args.output}", flush=True)


if __name__ == "__main__":
    main()
