import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .train import CachedRefinerDataset
from .model import DrumRefiner, RefinerConfig, apply_edits


def evaluate(ckpt_path: str, cache_dir: str, batch_size: int = 8):
    ds = CachedRefinerDataset(cache_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = DrumRefiner(RefinerConfig())
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_edit = 0.0
    total_vel = 0.0
    n = 0

    with torch.no_grad():
        for fm_logits, target, spec in dl:
            out = model(fm_logits, cond_feats=spec)
            refined = apply_edits(fm_logits, out["edit_logits"], out["vel_residual"], out["gate"])

            c = refined.shape[-1] // 2
            ref = refined.view(refined.size(0), refined.size(1), c, 2)
            tgt = target.view(target.size(0), target.size(1), c, 2)

            edit_err = (ref[..., 0].sign() != tgt[..., 0].sign()).float().mean()
            hit_mask = (tgt[..., 0] > 0).float()
            vel_mae = (torch.abs(ref[..., 1] - tgt[..., 1]) * hit_mask).sum() / (hit_mask.sum() + 1e-6)

            total_edit += float(edit_err)
            total_vel += float(vel_mae)
            n += 1

    print(f"edit_error={total_edit/max(1,n):.4f}, vel_mae={total_vel/max(1,n):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache_dir", default="./refiner/cache")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    evaluate(args.ckpt, args.cache_dir, args.batch_size)
