import argparse

import numpy as np
import torch

from src.config import Config as FMConfig
from .model import DrumRefiner, RefinerConfig, apply_edits

DRUM_MAPPING = [36, 38, 42, 47, 49, 51, 56]


def logits_to_events(refined_logits: torch.Tensor, fps: int = 100, threshold: float = 0.5):
    x = refined_logits.detach().cpu().numpy()
    if x.ndim == 3:
        x = x[0]

    c = x.shape[-1] // 2
    view = x.reshape(x.shape[0], c, 2)
    onset = view[..., 0]
    vel = np.clip(((view[..., 1] + 1.0) / 2.0) * 127.0, 1, 127)

    events = []
    for ch in range(c):
        frames = np.where(onset[:, ch] > threshold)[0]
        for fr in frames:
            events.append(
                {
                    "time": float(fr / fps),
                    "drum": int(ch),
                    "midi_note": int(DRUM_MAPPING[ch]),
                    "velocity": int(vel[fr, ch]),
                    "onset_score": float(onset[fr, ch]),
                }
            )
    return sorted(events, key=lambda x: x["time"])


def main():
    p = argparse.ArgumentParser(description="Run drum refiner from cached FM logits")
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--fm_logits_pt", required=True, help="tensor path of shape (T, C*2) or (1,T,C*2)")
    p.add_argument("--strength", type=float, default=1.0, help="manual override for refine strength")
    args = p.parse_args()

    fm_cfg = FMConfig()
    model = DrumRefiner(RefinerConfig(drum_channels=fm_cfg.DRUM_CHANNELS))
    ckpt = torch.load(args.refiner_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    fm_logits = torch.load(args.fm_logits_pt, map_location="cpu")
    if isinstance(fm_logits, dict):
        fm_logits = fm_logits["fm_logits"]
    if fm_logits.ndim == 2:
        fm_logits = fm_logits.unsqueeze(0)

    with torch.no_grad():
        out = model(fm_logits, cond_feats=None)
        out["gate"] = torch.clamp(out["gate"] * args.strength, 0.0, 1.0)
        refined = apply_edits(fm_logits, out["edit_logits"], out["vel_residual"], out["gate"])

    events = logits_to_events(refined, fps=fm_cfg.FPS)
    print(f"num_events={len(events)}")
    print(events[:20])


if __name__ == "__main__":
    main()
