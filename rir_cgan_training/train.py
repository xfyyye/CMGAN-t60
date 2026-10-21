import argparse
import json
from pathlib import Path

import torch
from torch import autograd
from torch.utils.data import DataLoader

from dataset import RIRDataset
from models import Critic, Generator


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gradient_penalty(critic, real, fake, t60):
    alpha = torch.rand(real.size(0), 1, 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(mixed, t60)
    grad = autograd.grad(score, mixed, torch.ones_like(score), create_graph=True,
                         retain_graph=True, only_inputs=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def main(args):
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out_dir); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    ds = RIRDataset(args.manifest, cfg["sample_rate"], cfg["rir_length"], "train",
                    cfg["t60_min"], cfg["t60_max"])
    loader = DataLoader(ds, cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], drop_last=True)
    g = Generator(cfg["z_dim"], cfg["cond_dim"], cfg["base_channels"], cfg["rir_length"]).to(device)
    d = Critic(cfg["cond_dim"], cfg["base_channels"], cfg["rir_length"]).to(device)
    opt_g = torch.optim.Adam(g.parameters(), cfg["lr"], (cfg["beta1"], cfg["beta2"]))
    opt_d = torch.optim.Adam(d.parameters(), cfg["lr"], (cfg["beta1"], cfg["beta2"]))
    step = 0
    for epoch in range(cfg["epochs"]):
        for batch in loader:
            real, t60 = batch["rir"].to(device), batch["t60"].to(device)
            z = torch.randn(real.size(0), cfg["z_dim"], device=device)
            fake = g(z, t60).detach()
            loss_d = d(fake, t60).mean() - d(real, t60).mean()
            loss_d = loss_d + cfg["gradient_penalty"] * gradient_penalty(d, real, fake, t60)
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
            if step % cfg["n_critic"] == 0:
                z = torch.randn(real.size(0), cfg["z_dim"], device=device)
                loss_g = -d(g(z, t60), t60).mean()
                opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
            if step % cfg["log_interval"] == 0:
                print(f"epoch={epoch} step={step} d={loss_d.item():.4f} g={loss_g.item():.4f}", flush=True)
            step += 1
        state = {"generator": g.state_dict(), "critic": d.state_dict(),
                 "config": cfg, "epoch": epoch, "step": step}
        torch.save(state, out / "checkpoints" / "latest.pt")
        if (epoch + 1) % cfg["save_interval"] == 0:
            torch.save(state, out / "checkpoints" / f"epoch_{epoch + 1:04d}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", default="config.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default=None)
    main(p.parse_args())

