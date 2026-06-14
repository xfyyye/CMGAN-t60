"""
隐藏层表征能力分析脚本
Usage:
    python analyze_representations.py [--gpu 2] [--n_samples 500] [--out_dir figures/representation]

分析方法（均在 test1 子集上进行）:
    1. 线性探针  — 冻结 backbone，各层输出接 Ridge 回归测 T60，汇报 R²
    2. t-SNE    — 各层输出降维，按 T60 连续值着色，对比单任务/多任务聚类结构
    3. CKA      — 单任务 vs 多任务各层特征的线性 CKA 相似度
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dataset import CMGANT60Dataset
from models.generator_t60 import TSCNet_T60Estimator_2TSCB, TSCNet_KAN_MultiTask_2TSCB

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

DATASET_ROOT = "/mnt/tidal-sh01/usr/chuan/youling/project/dataset/T60_Dataset_v7_4s"
N_FFT = 400
HOP = 100
SR = 16000
AUDIO_LEN = 4.0
T60_MIN, T60_MAX = 0.1, 1.5

MODELS = {
    "single_task": {
        "ckpt": ROOT / "runs/single_t60_kan_v2_mae_clip1_log/best_model.pth",
        "label": "Single-task",
        "color": "#9E9E9E",
        "cls": "single",
    },
    "w_denoise": {
        "ckpt": ROOT / "runs/kan_multitask_w_denoise/best_model.pth",
        "label": r"Multi-task w_denoise ($\alpha$=1.0, $\beta$=0.1)",
        "color": "#2196F3",
        "cls": "multi",
    },
    "w_balanced": {
        "ckpt": ROOT / "runs/kan_multitask_w_balanced/best_model.pth",
        "label": r"Multi-task w_balanced ($\alpha$=0.5, $\beta$=0.5)",
        "color": "#4CAF50",
        "cls": "multi",
    },
    "w_t60": {
        "ckpt": ROOT / "runs/kan_multitask_w_t60/best_model.pth",
        "label": r"Multi-task w_t60 ($\alpha$=0.1, $\beta$=1.0)",
        "color": "#F44336",
        "cls": "multi",
    },
}

LAYER_NAMES = ["encoder", "tscb_1", "tscb_2"]
T60_RMSE = {
    "single_task": 107.70,
    "w_denoise": 96.41,
    "w_balanced": 100.28,
    "w_eq": 105.64,
    "w_t60": 107.84,
}


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def load_model(cfg, device):
    if cfg["cls"] == "single":
        model = TSCNet_T60Estimator_2TSCB(
            t60_head_type="fourier_kan",
            t60_fourier_proj_dim=64,
            t60_fourier_hidden_dims=(32, 16),
            t60_fourier_first_num_frequencies=16,
            t60_fourier_hidden_num_frequencies=8,
            t60_fourier_dropout=0.1,
            t60_out_activation="sigmoid",
        )
    else:
        model = TSCNet_KAN_MultiTask_2TSCB(
            t60_head_type="fourier_kan",
            t60_fourier_proj_dim=64,
            t60_fourier_hidden_dims=(32, 16),
            t60_fourier_first_num_frequencies=16,
            t60_fourier_hidden_num_frequencies=8,
            t60_fourier_dropout=0.1,
            t60_out_activation="sigmoid",
        )
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ── 特征提取（各层 hook） ────────────────────────────────────────────────────

def extract_features(model, loader, device, n_samples):
    """
    返回 dict: {
        'encoder': (N, C),   # 对 (B,C,T,F) 做 mean(T,F)
        'tscb_1':  (N, C),
        'tscb_2':  (N, C),
        't60_raw': (N,),
    }
    """
    feats = {l: [] for l in LAYER_NAMES}
    t60s = []
    hooks = []
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            # output: (B, C, T, F) → mean over T,F → (B, C)
            activations[name] = output.detach().cpu().float().mean(dim=(-1, -2))
        return hook

    hooks.append(model.dense_encoder.register_forward_hook(make_hook("encoder")))
    hooks.append(model.TSCB_1.register_forward_hook(make_hook("tscb_1")))
    hooks.append(model.TSCB_2.register_forward_hook(make_hook("tscb_2")))

    collected = 0
    with torch.no_grad():
        for batch in loader:
            if collected >= n_samples:
                break
            x = batch["noisy_spec"].to(device)          # (B, 2, T, F)
            t60_raw = batch["t60_raw"].numpy()

            # 前向触发 hooks（多任务模型有两个输出，用 _ 接收）
            _ = model(x)

            for l in LAYER_NAMES:
                if l in activations:
                    feats[l].append(activations[l].numpy())
            t60s.append(t60_raw)
            collected += x.shape[0]

    for h in hooks:
        h.remove()

    result = {l: np.concatenate(feats[l], axis=0)[:n_samples] for l in LAYER_NAMES}
    result["t60_raw"] = np.concatenate(t60s, axis=0)[:n_samples]
    return result


# ── 线性探针 ─────────────────────────────────────────────────────────────────

def linear_probe(all_feats):
    """
    对每个模型的每层特征，做 80/20 split + Ridge 回归预测 T60，返回 R²。
    返回: dict {model_key: {layer: r2}}
    """
    results = {}
    for model_key, feats in all_feats.items():
        results[model_key] = {}
        t60 = feats["t60_raw"]
        n = len(t60)
        idx = np.random.permutation(n)
        train_idx = idx[:int(n * 0.8)]
        test_idx  = idx[int(n * 0.8):]
        for layer in LAYER_NAMES:
            X = feats[layer]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test  = scaler.transform(X[test_idx])
            reg = Ridge(alpha=1.0)
            reg.fit(X_train, t60[train_idx])
            y_pred = reg.predict(X_test)
            r2 = r2_score(t60[test_idx], y_pred)
            results[model_key][layer] = r2
            print(f"    [{model_key}] {layer}: R²={r2:.4f}")
    return results


def plot_linear_probe(probe_results, out_dir: Path):
    models_ordered = [m for m in ["single_task", "w_denoise", "w_balanced", "w_t60"]
                      if m in probe_results]
    x = np.arange(len(LAYER_NAMES))
    width = 0.8 / len(models_ordered)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model_key in enumerate(models_ordered):
        cfg = MODELS[model_key]
        r2s = [probe_results[model_key][l] for l in LAYER_NAMES]
        offset = (i - len(models_ordered)/2 + 0.5) * width
        bars = ax.bar(x + offset, r2s, width, color=cfg["color"],
                      label=cfg["label"], edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(["Encoder", "TSCB-1", "TSCB-2"])
    ax.set_ylabel("Linear Probe R² (T60 regression)")
    ax.set_title("Linear Probe: T60 Decodability from Each Layer\n"
                 "(higher R² = more T60 information encoded)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = out_dir / "linear_probe.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── t-SNE ────────────────────────────────────────────────────────────────────

def plot_tsne(all_feats, out_dir: Path, layer="tscb_2", n_tsne=500):
    models_ordered = [m for m in ["single_task", "w_denoise", "w_balanced", "w_t60"]
                      if m in all_feats]
    n = len(models_ordered)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    print(f"  Running t-SNE on layer={layer} (n_samples={n_tsne})...")
    for ax, model_key in zip(axes, models_ordered):
        cfg = MODELS[model_key]
        X = all_feats[model_key][layer][:n_tsne]
        t60 = all_feats[model_key]["t60_raw"][:n_tsne]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_2d = tsne.fit_transform(X_scaled)

        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=t60, cmap="plasma",
                        s=12, alpha=0.75, vmin=T60_MIN, vmax=T60_MAX)
        plt.colorbar(sc, ax=ax, label="T60 (s)", shrink=0.85)
        ax.set_title(f"{model_key}\nRMSE={T60_RMSE.get(model_key, '?'):.1f} ms",
                     fontsize=10)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"t-SNE of {layer.upper()} Representations (colored by T60 value)\n"
                 "Smoother gradient = better T60 encoding structure",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / f"tsne_{layer}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── CKA ──────────────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """Linear Centered Kernel Alignment between X (N,d1) and Y (N,d2)."""
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K = X @ X.T
    L = Y @ Y.T
    Kc = center(K)
    Lc = center(L)
    num = np.sum(Kc * Lc)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return num / (denom + 1e-10)


def plot_cka(all_feats, out_dir: Path):
    models_ordered = [m for m in ["single_task", "w_denoise", "w_balanced", "w_t60"]
                      if m in all_feats]
    if "single_task" not in models_ordered:
        print("  CKA skipped: single_task features not available")
        return

    # CKA matrix: rows = layers, cols = models (vs single_task)
    multi_models = [m for m in models_ordered if m != "single_task"]
    cka_matrix = np.zeros((len(LAYER_NAMES), len(multi_models)))

    print("  Computing CKA...")
    single_feats = all_feats["single_task"]
    for j, model_key in enumerate(multi_models):
        multi_feats = all_feats[model_key]
        for i, layer in enumerate(LAYER_NAMES):
            X = single_feats[layer]
            Y = multi_feats[layer]
            n = min(len(X), len(Y))
            cka_val = linear_cka(X[:n], Y[:n])
            cka_matrix[i, j] = cka_val
            print(f"    CKA single_task vs {model_key} @ {layer}: {cka_val:.4f}")

    fig, ax = plt.subplots(figsize=(5 + len(multi_models), 4))
    im = ax.imshow(cka_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Linear CKA")
    ax.set_xticks(range(len(multi_models)))
    ax.set_xticklabels([m.replace("w_", "w_\n") for m in multi_models], fontsize=9)
    ax.set_yticks(range(len(LAYER_NAMES)))
    ax.set_yticklabels(["Encoder", "TSCB-1", "TSCB-2"])
    ax.set_xlabel("Multi-task Model")
    ax.set_ylabel("Layer")
    ax.set_title("Linear CKA: Single-task vs Multi-task Feature Similarity\n"
                 "(lower CKA = representations diverged more from single-task baseline)")
    for i in range(len(LAYER_NAMES)):
        for j in range(len(multi_models)):
            ax.text(j, i, f"{cka_matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if cka_matrix[i,j] < 0.4 else "black")
    fig.tight_layout()
    path = out_dir / "cka_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of test1 samples to use")
    parser.add_argument("--out_dir", default="figures/representation")
    parser.add_argument("--models", nargs="+",
                        default=["single_task", "w_denoise", "w_balanced", "w_t60"],
                        help="Which models to analyze")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据集
    dataset = CMGANT60Dataset(
        dataset_root=DATASET_ROOT,
        split="test1",
        n_fft=N_FFT,
        hop_length=HOP,
        audio_length=AUDIO_LEN,
        target_sr=SR,
        t60_range=(T60_MIN, T60_MAX),
    )
    n = min(args.n_samples, len(dataset))
    indices = np.random.default_rng(42).choice(len(dataset), n, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=4)
    print(f"Dataset: test1, using {n} samples")

    # 提取各模型特征
    all_feats = {}
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"  Unknown model: {model_key}, skipping")
            continue
        cfg = MODELS[model_key]
        if not cfg["ckpt"].exists():
            print(f"  Checkpoint not found: {cfg['ckpt']}, skipping {model_key}")
            continue
        print(f"\nExtracting features: {model_key}...")
        model = load_model(cfg, device)
        all_feats[model_key] = extract_features(model, loader, device, n)
        del model
        torch.cuda.empty_cache()

    if not all_feats:
        print("No features extracted. Exiting.")
        return

    # 线性探针
    print("\n[Linear Probe]")
    probe_results = linear_probe(all_feats)
    plot_linear_probe(probe_results, out_dir)

    # t-SNE（在 tscb_2 层，特征最丰富）
    print("\n[t-SNE]")
    plot_tsne(all_feats, out_dir, layer="tscb_2", n_tsne=min(500, n))
    # 也画 encoder 层（最底层，看早期特征差异）
    plot_tsne(all_feats, out_dir, layer="encoder", n_tsne=min(500, n))

    # CKA
    print("\n[CKA]")
    plot_cka(all_feats, out_dir)

    print(f"\nDone. All figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
