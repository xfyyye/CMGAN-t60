import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from models import Generator
from rir_metrics import accept_rir


def main(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    g = Generator(cfg["z_dim"], cfg["cond_dim"], cfg["base_channels"], cfg["rir_length"]).to(device)
    g.load_state_dict(ckpt["generator"]); g.eval()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Install soundfile before generation") from exc
    records = []
    with torch.no_grad():
        for target in args.t60:
            folder = out / f"t60_{target:.3f}"; folder.mkdir(exist_ok=True)
            kept = 0; attempts = 0
            while kept < args.num_per_t60 and attempts < args.num_per_t60 * args.max_attempt_factor:
                n = min(args.batch_size, args.num_per_t60 - kept + args.batch_size)
                z = torch.randn(n, cfg["z_dim"], device=device)
                c = torch.full((n, 1), (target - cfg["t60_min"]) / (cfg["t60_max"] - cfg["t60_min"]), device=device)
                batch = g(z, c).cpu().numpy()[:, 0]
                for rir in batch:
                    attempts += 1
                    info = accept_rir(rir, target, cfg["sample_rate"], cfg["t60_tolerance_sec"], cfg["min_rms"], cfg["max_peak"])
                    if not info["accepted"]: continue
                    name = f"rir_t60_{target:.3f}_{kept:05d}.wav"
                    sf.write(folder / name, rir, cfg["sample_rate"])
                    records.append({"path": str(folder / name), "target_t60": target, **info})
                    kept += 1
                    if kept >= args.num_per_t60: break
            print(f"target={target:.3f} kept={kept}/{args.num_per_t60} attempts={attempts}")
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["path", "target_t60", "measured_t60", "t60_error", "peak", "rms", "accepted"]
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="config.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--t60", nargs="+", type=float, required=True)
    p.add_argument("--num_per_t60", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_attempt_factor", type=int, default=10)
    p.add_argument("--device", default=None)
    main(p.parse_args())

