import argparse
import json
import os
from collections import defaultdict

import mir_eval
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Config as FMConfig
from src.dataset import EGMDEvalDataset
from src.model import AnnealedPseudoHuberLoss, FlowMatchingTransformer

from .config import RefinerTrainConfig
from .model import DrumRefiner, RefinerConfig, apply_edits

DRUM_NAMES = ["Kick", "Snare", "HH", "Toms", "Crash", "Ride", "Bell"]


class CachedRefinerEvalDataset(Dataset):
    def __init__(self, root: str):
        from pathlib import Path

        self.paths = sorted(Path(root).glob("sample_*.pt"))
        if not self.paths:
            raise RuntimeError(f"No cached samples in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        obj = torch.load(self.paths[idx], map_location="cpu")
        return {
            "fm_logits": obj["fm_logits"],
            "target": obj["target"],
            "spec": obj["spec"],
            "file_id": obj["file_id"],
            "start_time": float(obj["start_time"]),
            "end_time": float(obj["end_time"]),
            "fm_loss": float(obj.get("fm_loss", 0.0)),
        }


def _resolve_devices(max_gpus: int = 4):
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    n = min(torch.cuda.device_count(), max_gpus)
    ids = list(range(n))
    return torch.device(f"cuda:{ids[0]}"), ids


def precompute_eval_cache(args, device, device_ids):
    os.makedirs(args.cache_dir, exist_ok=True)

    train_cfg = RefinerTrainConfig()
    fm_cfg = FMConfig()
    fm_cfg.SEGMENT_SEC = train_cfg.segment_sec

    ds = EGMDEvalDataset(overlap_ratio=0.5, limit=args.num_samples, segment_len=train_cfg.segment_sec)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=fm_cfg.NUM_WORKERS)

    fm = FlowMatchingTransformer(fm_cfg).to(device)
    loss_wrap = AnnealedPseudoHuberLoss(fm, fm_cfg).to(device)

    if len(device_ids) > 1:
        print(f"[Refiner eval precompute] Using {len(device_ids)} GPUs: {device_ids}")
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

    with torch.no_grad():
        for idx, (wav_mert, spec, target) in enumerate(tqdm(dl, desc="Precompute eval FM logits")):
            wav_mert = wav_mert.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            fm_loss = loss_wrap(wav_mert, spec, target, progress=0.5).mean().item()
            fm_logits = loss_wrap.sample(wav_mert, spec, steps=args.eval_steps)

            seg_info = ds.files[idx]
            out = {
                "fm_logits": fm_logits[0].cpu(),
                "target": target[0].cpu(),
                "spec": spec[0].cpu(),
                "file_id": seg_info["file_id"],
                "start_time": seg_info["start_time"],
                "end_time": seg_info["end_time"],
                "fm_loss": fm_loss,
            }
            torch.save(out, os.path.join(args.cache_dir, f"sample_{idx:08d}.pt"))


class RefinerEvaluator:
    def __init__(self, args):
        self.args = args
        self.config = FMConfig()

    def stitch_predictions(self, segment_results):
        file_predictions = {}
        file_targets = {}

        files_map = defaultdict(list)
        for result in segment_results:
            files_map[result["file_id"]].append(result)

        for file_id, segments in files_map.items():
            segments.sort(key=lambda x: x["start_time"])
            max_end_time = max([s["end_time"] for s in segments])
            grid_length = int(max_end_time * self.config.FPS) + 1
            drum_channels = self.config.DRUM_CHANNELS

            stitched_pred = np.full((grid_length, drum_channels * 2), -1.0, dtype=np.float32)
            stitched_target = np.full((grid_length, drum_channels * 2), -1.0, dtype=np.float32)
            overlap_count = np.zeros(grid_length, dtype=int)

            for segment in segments:
                start_frame = int(segment["start_time"] * self.config.FPS)
                pred_grid = segment["pred_grid"]
                target_grid = segment["target_grid"]

                segment_length = min(pred_grid.shape[0], grid_length - start_frame)
                if segment_length <= 0:
                    continue

                for frame_idx in range(segment_length):
                    global_frame = start_frame + frame_idx
                    if overlap_count[global_frame] == 0:
                        stitched_pred[global_frame] = pred_grid[frame_idx]
                        stitched_target[global_frame] = target_grid[frame_idx]
                    else:
                        alpha = 1.0 / (overlap_count[global_frame] + 1)
                        stitched_pred[global_frame] = (1 - alpha) * stitched_pred[global_frame] + alpha * pred_grid[frame_idx]
                        if np.all(stitched_target[global_frame] == -1.0) and not np.all(target_grid[frame_idx] == -1.0):
                            stitched_target[global_frame] = target_grid[frame_idx]
                    overlap_count[global_frame] += 1

            file_predictions[file_id] = stitched_pred
            file_targets[file_id] = stitched_target

        return file_predictions, file_targets

    def calculate_onset_velocity_metrics(self, ref_intervals, ref_pitches, ref_vels, est_intervals, est_pitches, est_vels):
        try:
            if len(est_intervals) == 0 or len(ref_intervals) == 0:
                return 0.0, 0.0, 0.0

            matched_pairs = mir_eval.transcription.match_notes(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                onset_tolerance=0.05,
                pitch_tolerance=0.0,
                offset_ratio=None,
            )

            velocity_tp = 0
            vel_tolerance = 12.7
            for ref_idx, est_idx in matched_pairs:
                if abs(ref_vels[ref_idx] - est_vels[est_idx]) <= vel_tolerance:
                    velocity_tp += 1

            n_ref = len(ref_vels)
            n_est = len(est_vels)
            precision = velocity_tp / n_est if n_est > 0 else 0.0
            recall = velocity_tp / n_ref if n_ref > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return precision, recall, f1
        except Exception as e:
            print(f"Warning: Error in onset_velocity_metrics calculation: {e}")
            return 0.0, 0.0, 0.0

    def calculate_onset_only_metrics(self, ref_intervals, ref_pitches, est_intervals, est_pitches):
        try:
            if len(ref_intervals) == 0 or len(est_intervals) == 0:
                return 0.0, 0.0, 0.0
            prec, rec, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                onset_tolerance=0.05,
                pitch_tolerance=0.0,
                offset_ratio=None,
            )
            return float(prec), float(rec), float(f1)
        except Exception as e:
            print(f"Warning: Error in onset-only metrics calculation: {e}")
            return 0.0, 0.0, 0.0

    def calculate_file_metrics(self, pred_grid, target_grid):
        drum_channels = self.config.DRUM_CHANNELS
        pred_view = pred_grid.reshape(pred_grid.shape[0], drum_channels, 2)
        pred_onset = pred_view[:, :, 0]
        pred_velocity = pred_view[:, :, 1]

        target_view = target_grid.reshape(target_grid.shape[0], drum_channels, 2)
        target_onset = target_view[:, :, 0]
        target_velocity = target_view[:, :, 1]

        pred_notes = []
        target_notes = []
        frame_to_sec = 1.0 / self.config.FPS

        def denorm_vel(val):
            return np.clip(((val + 1) / 2) * 127, 1, 127)

        for drum_idx in range(drum_channels):
            pred_peaks, _ = find_peaks(pred_onset[:, drum_idx], height=0.0, distance=3)
            for peak_frame in pred_peaks:
                pred_notes.append([peak_frame * frame_to_sec, drum_idx + 1, denorm_vel(pred_velocity[peak_frame, drum_idx])])

            target_peaks = np.where(target_onset[:, drum_idx] > -0.5)[0]
            for peak_frame in target_peaks:
                target_notes.append([peak_frame * frame_to_sec, drum_idx + 1, denorm_vel(target_velocity[peak_frame, drum_idx])])

        pred_arr = np.array(pred_notes) if pred_notes else np.empty((0, 3))
        target_arr = np.array(target_notes) if target_notes else np.empty((0, 3))

        if len(pred_arr) > 0:
            p_intervals = np.column_stack([pred_arr[:, 0], pred_arr[:, 0] + 0.1])
            p_pitches = pred_arr[:, 1].astype(int)
            p_vels = pred_arr[:, 2]
        else:
            p_intervals, p_pitches, p_vels = np.empty((0, 2)), np.array([]), np.array([])

        if len(target_arr) > 0:
            t_intervals = np.column_stack([target_arr[:, 0], target_arr[:, 0] + 0.1])
            t_pitches = target_arr[:, 1].astype(int)
            t_vels = target_arr[:, 2]
        else:
            t_intervals, t_pitches, t_vels = np.empty((0, 2)), np.array([]), np.array([])

        o_prec, o_rec, o_f1 = self.calculate_onset_only_metrics(t_intervals, t_pitches, p_intervals, p_pitches)
        v_prec, v_rec, v_f1 = self.calculate_onset_velocity_metrics(t_intervals, t_pitches, t_vels, p_intervals, p_pitches, p_vels)

        per_drum = {}
        per_drum_onset_only = {}
        for i, drum_name in enumerate(DRUM_NAMES):
            pitch_id = i + 1
            p_mask = p_pitches == pitch_id if len(p_pitches) > 0 else np.array([], dtype=bool)
            t_mask = t_pitches == pitch_id if len(t_pitches) > 0 else np.array([], dtype=bool)

            sub_p_intervals = p_intervals[p_mask] if len(p_pitches) > 0 else np.empty((0, 2))
            sub_p_pitches = p_pitches[p_mask] if len(p_pitches) > 0 else np.array([])
            sub_p_vels = p_vels[p_mask] if len(p_pitches) > 0 else np.array([])

            sub_t_intervals = t_intervals[t_mask] if len(t_pitches) > 0 else np.empty((0, 2))
            sub_t_pitches = t_pitches[t_mask] if len(t_pitches) > 0 else np.array([])
            sub_t_vels = t_vels[t_mask] if len(t_pitches) > 0 else np.array([])

            d_prec, d_rec, d_f1 = self.calculate_onset_velocity_metrics(
                sub_t_intervals, sub_t_pitches, sub_t_vels, sub_p_intervals, sub_p_pitches, sub_p_vels
            )
            do_prec, do_rec, do_f1 = self.calculate_onset_only_metrics(
                sub_t_intervals, sub_t_pitches, sub_p_intervals, sub_p_pitches
            )

            per_drum[drum_name] = {"precision": d_prec, "recall": d_rec, "f1_score": d_f1}
            per_drum_onset_only[drum_name] = {"precision": do_prec, "recall": do_rec, "f1_score": do_f1}

        return {
            "onset_only": {"precision": o_prec, "recall": o_rec, "f1_score": o_f1},
            "onset_velocity": {"precision": v_prec, "recall": v_rec, "f1_score": v_f1},
            "per_drum": per_drum,
            "per_drum_onset_only": per_drum_onset_only,
        }

    @torch.no_grad()
    def evaluate(self, args):
        ds = CachedRefinerEvalDataset(args.cache_dir)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = DrumRefiner(RefinerConfig(drum_channels=self.config.DRUM_CHANNELS), cond_dim=self.config.N_MELS)
        ckpt = torch.load(args.refiner_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        segment_results = []
        all_losses = []

        for batch in tqdm(dl, desc="Refiner eval"):
            fm_logits = batch["fm_logits"]
            target = batch["target"]
            spec = batch["spec"]
            out = model(fm_logits, cond_feats=spec)
            refined = apply_edits(fm_logits, out["edit_logits"], out["vel_residual"], out["gate"])

            for i in range(refined.size(0)):
                segment_results.append(
                    {
                        "file_id": batch["file_id"][i],
                        "start_time": float(batch["start_time"][i]),
                        "end_time": float(batch["end_time"][i]),
                        "pred_grid": refined[i].cpu().numpy(),
                        "target_grid": target[i].cpu().numpy(),
                    }
                )
                all_losses.append(float(batch["fm_loss"][i]))

        file_predictions, file_targets = self.stitch_predictions(segment_results)

        file_metrics = {}
        for file_id in file_predictions:
            file_metrics[file_id] = self.calculate_file_metrics(file_predictions[file_id], file_targets[file_id])

        onset_avg = {
            "precision": float(np.mean([file_metrics[f]["onset_only"]["precision"] for f in file_metrics])),
            "recall": float(np.mean([file_metrics[f]["onset_only"]["recall"] for f in file_metrics])),
            "f1_score": float(np.mean([file_metrics[f]["onset_only"]["f1_score"] for f in file_metrics])),
        }
        velocity_avg = {
            "precision": float(np.mean([file_metrics[f]["onset_velocity"]["precision"] for f in file_metrics])),
            "recall": float(np.mean([file_metrics[f]["onset_velocity"]["recall"] for f in file_metrics])),
            "f1_score": float(np.mean([file_metrics[f]["onset_velocity"]["f1_score"] for f in file_metrics])),
        }

        per_drum_avg = {}
        per_drum_onset_only_avg = {}
        for drum_name in DRUM_NAMES:
            per_drum_avg[drum_name] = {
                "precision": float(np.mean([file_metrics[f]["per_drum"][drum_name]["precision"] for f in file_metrics])),
                "recall": float(np.mean([file_metrics[f]["per_drum"][drum_name]["recall"] for f in file_metrics])),
                "f1_score": float(np.mean([file_metrics[f]["per_drum"][drum_name]["f1_score"] for f in file_metrics])),
            }
            per_drum_onset_only_avg[drum_name] = {
                "precision": float(np.mean([file_metrics[f]["per_drum_onset_only"][drum_name]["precision"] for f in file_metrics])),
                "recall": float(np.mean([file_metrics[f]["per_drum_onset_only"][drum_name]["recall"] for f in file_metrics])),
                "f1_score": float(np.mean([file_metrics[f]["per_drum_onset_only"][drum_name]["f1_score"] for f in file_metrics])),
            }

        results = {
            "model_info": {"checkpoint_epoch": ckpt.get("epoch", "Unknown"), "evaluation_mode": "quick"},
            "summary": {
                "avg_loss": float(np.mean(all_losses)) if all_losses else 0.0,
                "num_files": len(file_metrics),
                "num_segments": len(segment_results),
                "onset_only_average": onset_avg,
                "onset_velocity_average": velocity_avg,
                "per_drum_average": per_drum_avg,
                "per_drum_onset_only_average": per_drum_onset_only_avg,
            },
            "per_file_metrics": file_metrics,
        }

        print(f"\n🚀 Quick Evaluation Results:")
        print(f"   Average Loss: {results['summary']['avg_loss']:.4f}")
        print(f"   Total Files: {results['summary']['num_files']}")
        print(f"   Total Segments Processed: {results['summary']['num_segments']}")
        print(f"\n📁 Onset-Only Metrics (mean across {len(file_metrics)} files):")
        print(f"   Overall: P={onset_avg['precision']:.3f} R={onset_avg['recall']:.3f} F1={onset_avg['f1_score']:.3f}")
        print(f"\n📁 Onset+Velocity Metrics (mean across {len(file_metrics)} files):")
        print(f"   Overall: P={velocity_avg['precision']:.3f} R={velocity_avg['recall']:.3f} F1={velocity_avg['f1_score']:.3f}")
        print(f"\n🥁 Per-Drum Onset+Velocity:")
        for drum_name in DRUM_NAMES:
            m = per_drum_avg[drum_name]
            print(f"   {drum_name:8}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1_score']:.3f}")
        print(f"\n🥁 Per-Drum Onset Only (50ms):")
        for drum_name in DRUM_NAMES:
            m = per_drum_onset_only_avg[drum_name]
            print(f"   {drum_name:8}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1_score']:.3f}")

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"quick_eval_ep{ckpt.get('epoch', 'Unknown')}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Refiner evaluation on cached FM logits")
    p.add_argument("--refiner_ckpt", required=True, help="Path to refiner checkpoint")
    p.add_argument("--cache_dir", default="./refiner/eval_cache", help="Path to eval cache directory")
    p.add_argument("--output_dir", default="./refiner/eval_results", help="Directory for evaluation json output")
    p.add_argument("--batch_size", type=int, default=8)

    p.add_argument("--precache", action="store_true", help="Precompute FM logits on EGMD eval set (200 segments)")
    p.add_argument("--fm_ckpt", type=str, default="", help="FM checkpoint path (required with --precache)")
    p.add_argument("--eval_steps", type=int, default=5, help="FM sampling steps for caching")
    p.add_argument("--num_samples", type=int, default=200, help="Number of eval segments to cache")
    p.add_argument("--max_gpus", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if args.precache:
        if not args.fm_ckpt:
            raise ValueError("--fm_ckpt is required when --precache is set")
        device, device_ids = _resolve_devices(max_gpus=args.max_gpus)
        precompute_eval_cache(args, device, device_ids)

    evaluator = RefinerEvaluator(args)
    evaluator.evaluate(args)


if __name__ == "__main__":
    main()
