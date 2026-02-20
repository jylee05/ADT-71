import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Config as FMConfig
from src.dataset import EGMDTrainDataset
from src.model import AnnealedPseudoHuberLoss, FlowMatchingTransformer

from .config import RefinerTrainConfig
from .model import DrumRefiner, RefinerConfig, apply_edits, make_edit_labels


class CachedRefinerDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(Path(root).glob("sample_*.pt"))
        if not self.paths:
            raise RuntimeError(f"No cached samples in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        obj = torch.load(self.paths[idx], map_location="cpu")
        return obj["fm_logits"], obj["target"], obj["spec"]


def _resolve_devices(max_gpus: int = 4):
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    n = min(torch.cuda.device_count(), max_gpus)
    ids = list(range(n))
    return torch.device(f"cuda:{ids[0]}"), ids


def corrupt_like_noisy_audio(logits: torch.Tensor, cfg: RefinerTrainConfig) -> torch.Tensor:
    """Corruption that mimics noisy-audio FM artifacts (mostly add/delete)."""
    b, t, d = logits.shape
    c = d // 2
    view = logits.view(b, t, c, 2)
    on = view[..., 0]
    vel = view[..., 1]

    hit = on > 0
    add_mask = (~hit) & (torch.rand_like(on) < cfg.p_add)
    del_mask = hit & (torch.rand_like(on) < cfg.p_delete)

    on = torch.where(add_mask, torch.ones_like(on), on)
    on = torch.where(del_mask, -torch.ones_like(on), on)

    vel = torch.clamp(vel + torch.randn_like(vel) * cfg.vel_noise_std, -1.0, 1.0)
    return torch.stack([on, vel], dim=-1).reshape(b, t, d)


def estimate_uncertainty(fm_logits: torch.Tensor, margin: float = 0.75) -> torch.Tensor:
    c = fm_logits.shape[-1] // 2
    on = fm_logits.view(fm_logits.size(0), fm_logits.size(1), c, 2)[..., 0]
    return (on.abs() < margin).float().mean(dim=[1, 2], keepdim=True)


def precompute_fm_cache(args, train_cfg, fm_cfg, device, device_ids):
    os.makedirs(train_cfg.cache_dir, exist_ok=True)

    fm_cfg.SEGMENT_SEC = train_cfg.segment_sec
    ds = EGMDTrainDataset(compute_sample_weights=False)

    precompute_bs = args.precompute_batch_size if args.precompute_batch_size > 0 else fm_cfg.BATCH_SIZE
    dl = DataLoader(ds, batch_size=precompute_bs, shuffle=False, num_workers=fm_cfg.NUM_WORKERS)

    fm = FlowMatchingTransformer(fm_cfg).to(device)
    loss_wrap = AnnealedPseudoHuberLoss(fm, fm_cfg).to(device)

    if len(device_ids) > 1:
        print(f"[Refiner precompute] Using {len(device_ids)} GPUs: {device_ids}")
        fm = nn.DataParallel(fm, device_ids=device_ids)
        loss_wrap.model = fm

    ckpt = torch.load(args.fm_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    target_module = fm.module if isinstance(fm, nn.DataParallel) else fm
    target_module.load_state_dict(state, strict=True)
    target_module.eval()
    for p in target_module.parameters():
        p.requires_grad = False

    sample_id = 0
    with torch.no_grad():
        for wav_mert, spec, target in tqdm(dl, desc="Precompute FM logits"):
            wav_mert = wav_mert.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            fm_logits = loss_wrap.sample(wav_mert, spec, steps=args.fm_steps)
            for i in range(fm_logits.size(0)):
                out = {
                    "fm_logits": fm_logits[i].cpu(),
                    "target": target[i].cpu(),
                    "spec": spec[i].cpu(),
                }
                torch.save(out, os.path.join(train_cfg.cache_dir, f"sample_{sample_id:08d}.pt"))
                sample_id += 1


def train_refiner(args):
    device, device_ids = _resolve_devices(max_gpus=args.max_gpus)

    train_cfg = RefinerTrainConfig()
    fm_cfg = FMConfig()
    fm_cfg.SEGMENT_SEC = train_cfg.segment_sec

    print(f"[Refiner] device={device}, device_ids={device_ids if device_ids else 'cpu'}")
    print(f"[Refiner] batch_size(from FM config)={fm_cfg.BATCH_SIZE}, num_workers(from FM config)={fm_cfg.NUM_WORKERS}")

    if args.precompute_cache:
        precompute_fm_cache(args, train_cfg, fm_cfg, device, device_ids)

    ds = CachedRefinerDataset(train_cfg.cache_dir)
    dl = DataLoader(
        ds,
        batch_size=fm_cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=fm_cfg.NUM_WORKERS,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    model = DrumRefiner(
        RefinerConfig(drum_channels=fm_cfg.DRUM_CHANNELS),
        cond_dim=fm_cfg.N_MELS,
    ).to(device)
    if len(device_ids) > 1:
        print(f"[Refiner train] Using {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    os.makedirs(train_cfg.save_dir, exist_ok=True)

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running = 0.0
        for fm_logits, target, spec in tqdm(dl, desc=f"Epoch {epoch}"):
            fm_logits = fm_logits.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)

            if torch.rand(1).item() < train_cfg.p_extra_corrupt:
                fm_in = corrupt_like_noisy_audio(fm_logits, train_cfg)
            else:
                fm_in = fm_logits

            out = model(fm_in, cond_feats=spec)
            refined = apply_edits(fm_in, out["edit_logits"], out["vel_residual"], out["gate"])

            c = fm_cfg.DRUM_CHANNELS
            base_view = fm_in.view(fm_in.size(0), fm_in.size(1), c, 2)
            tgt_view = target.view(target.size(0), target.size(1), c, 2)
            ref_view = refined.view(refined.size(0), refined.size(1), c, 2)

            edit_labels = make_edit_labels(base_view[..., 0], tgt_view[..., 0])
            edit_loss = F.cross_entropy(out["edit_logits"].reshape(-1, 3), edit_labels.reshape(-1))

            hit_mask = (tgt_view[..., 0] > 0).float()
            vel_loss = (F.smooth_l1_loss(ref_view[..., 1], tgt_view[..., 1], reduction="none") * hit_mask).sum() / (hit_mask.sum() + 1e-6)

            uncertainty = estimate_uncertainty(fm_in, margin=train_cfg.uncertainty_margin)
            identity = ((refined - fm_in).abs().mean(dim=[1, 2], keepdim=True) * (1.0 - uncertainty)).mean()
            budget = (refined - fm_in).abs().mean()

            loss = (
                train_cfg.w_edit * edit_loss
                + train_cfg.w_vel * vel_loss
                + train_cfg.w_identity * identity
                + train_cfg.w_budget * budget
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = model.module.parameters() if isinstance(model, nn.DataParallel) else model.parameters()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            running += float(loss.item())

        avg = running / max(1, len(dl))
        print(f"Epoch {epoch}: loss={avg:.4f}")
        save_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({"model": save_model.state_dict(), "epoch": epoch}, os.path.join(train_cfg.save_dir, f"refiner_epoch_{epoch:03d}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-stage drum refiner training (FM frozen)")
    parser.add_argument("--fm_ckpt", type=str, required=True, help="Path to frozen FM checkpoint")
    parser.add_argument("--fm_steps", type=int, default=3, help="FM sampling steps used to generate logits")
    parser.add_argument("--precompute_cache", action="store_true", help="Generate and save FM logits cache first")
    parser.add_argument("--precompute_batch_size", type=int, default=0, help="If 0, use src.config.Config.BATCH_SIZE")
    parser.add_argument("--max_gpus", type=int, default=4, help="Max number of GPUs to use for both precompute and refiner train")

    args = parser.parse_args()
    train_refiner(args)
