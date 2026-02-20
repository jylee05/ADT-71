#!/usr/bin/env python3
"""
N2N-Flow2 Evaluation Script with Proper Stitching
Fixes overlap inflation by stitching predictions first, then evaluating
[Modified] Fixed F1 calculation logic using mir_eval
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from collections import defaultdict
from scipy.signal import find_peaks
import mir_eval  # [수정] 정확한 매칭을 위해 라이브러리 추가

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDEvalDataset

# Drum class names for reporting
DRUM_NAMES = ["Kick", "Snare", "HH", "Toms", "Crash", "Ride", "Bell"]

class QuickEGMDEvalDataset(EGMDEvalDataset):
    """Quick evaluation dataset with random 200 samples"""
    def __init__(self, num_samples=200, overlap_ratio=0.5):
        # 부모 생성자에게 limit를 전달하여, 로딩 단계에서부터 200개만 가져오게 함
        super().__init__(overlap_ratio=overlap_ratio, limit=num_samples)

class ProperEvaluator:
    def __init__(self, args):
        # Force GPU 1 usage for Docker environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        
        print(f"🎯 N2N-Flow2 Proper Evaluation (Stitch -> Evaluate)")
        print(f"   Device: {self.device}")
        print(f"   Evaluation Steps: {args.eval_steps}")
        
        # Load model
        print(f"📂 Loading model from: {args.ckpt_path}")
        self.model = FlowMatchingTransformer(self.config).to(self.device)
        self.loss_fn = AnnealedPseudoHuberLoss(self.model, self.config).to(self.device)
        self.load_checkpoint(args.ckpt_path)
        
        self.model.eval()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Extract epoch info (already 1-indexed in train.py)
        self.epoch = checkpoint.get('epoch', 'Unknown')
        self.train_loss = checkpoint.get('train_loss', checkpoint.get('loss', 'Unknown'))
        
        # Handle DataParallel checkpoints
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Model loaded successfully")
        print(f"   - Checkpoint epoch: {self.epoch}")

    def stitch_predictions(self, segment_results):
        """Stitch overlapping segment predictions into full-length files"""
        file_predictions = {}
        file_targets = {}
        
        # Group by file
        files_map = defaultdict(list)
        for result in segment_results:
            file_id = result['file_id']
            files_map[file_id].append(result)
        
        for file_id, segments in files_map.items():
            # Sort by start time
            segments.sort(key=lambda x: x['start_time'])
            
            # Find the maximum time to determine grid length
            max_end_time = max([s['end_time'] for s in segments])
            grid_length = int(max_end_time * self.config.FPS) + 1
            drum_channels = self.config.DRUM_CHANNELS
            
            # Initialize stitched grids
            stitched_pred = np.full((grid_length, drum_channels * 2), -1.0, dtype=np.float32)
            stitched_target = np.full((grid_length, drum_channels * 2), -1.0, dtype=np.float32)
            overlap_count = np.zeros(grid_length, dtype=int)
            
            # Stitch segments with proper overlap handling
            for segment in segments:
                start_frame = int(segment['start_time'] * self.config.FPS)
                pred_grid = segment['pred_grid']
                target_grid = segment['target_grid']
                
                segment_length = min(pred_grid.shape[0], grid_length - start_frame)
                end_frame = start_frame + segment_length
                
                if segment_length > 0:
                    # For overlap regions, average the predictions
                    for frame_idx in range(segment_length):
                        global_frame = start_frame + frame_idx
                        
                        if overlap_count[global_frame] == 0:
                            # First time seeing this frame
                            stitched_pred[global_frame] = pred_grid[frame_idx]
                            stitched_target[global_frame] = target_grid[frame_idx]
                        else:
                            # Average with existing predictions in overlap region
                            alpha = 1.0 / (overlap_count[global_frame] + 1)
                            stitched_pred[global_frame] = (1 - alpha) * stitched_pred[global_frame] + alpha * pred_grid[frame_idx]
                            # Target should be the same, but take the first non-empty one
                            if np.all(stitched_target[global_frame] == -1.0) and not np.all(target_grid[frame_idx] == -1.0):
                                stitched_target[global_frame] = target_grid[frame_idx]
                        
                        overlap_count[global_frame] += 1
            
            file_predictions[file_id] = stitched_pred
            file_targets[file_id] = stitched_target
            
            print(f"   Stitched {file_id}: {len(segments)} segments -> {grid_length} frames")
        
        return file_predictions, file_targets

    def calculate_onset_velocity_metrics(self, ref_intervals, ref_pitches, ref_vels, est_intervals, est_pitches, est_vels):
        """Calculate Precision, Recall, F1 using 50ms Onset + Velocity"""
        # [수정] mir_eval의 match_notes를 사용하여 최적 매칭(Bipartite Matching) 수행
        # 기존의 단순 for loop는 중복 매칭 오류가 있었음
        try:
            if len(est_intervals) == 0 or len(ref_intervals) == 0:
                return 0.0, 0.0, 0.0
                
            # 1. Onset 기준으로 노트 매칭 (50ms tolerance)
            matched_pairs = mir_eval.transcription.match_notes(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                onset_tolerance=0.05,
                pitch_tolerance=0.0,
                offset_ratio=None
            )
            
            # 2. 매칭된 쌍 중에서 Velocity 조건 확인
            velocity_tp = 0
            VEL_TOLERANCE = 12.7  # 10% of 127
            
            for ref_idx, est_idx in matched_pairs:
                r_vel = ref_vels[ref_idx]
                e_vel = est_vels[est_idx]
                if abs(r_vel - e_vel) <= VEL_TOLERANCE:
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

    # ✅ [추가] per-drum onset-only 계산에도 재사용할 수 있게 함수로 분리(기존 로직과 동일 mir_eval 호출)
    def calculate_onset_only_metrics(self, ref_intervals, ref_pitches, est_intervals, est_pitches):
        """Calculate Precision, Recall, F1 using 50ms Onset Only (No Velocity)"""
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
                offset_ratio=None
            )
            return float(prec), float(rec), float(f1)
        except Exception as e:
            print(f"Warning: Error in onset-only metrics calculation: {e}")
            return 0.0, 0.0, 0.0

    def calculate_file_metrics(self, pred_grid, target_grid, time_offset=0.0):
        """Calculate note-level metrics for a complete file after stitching"""
        drum_channels = self.config.DRUM_CHANNELS
        
        # Split predictions and targets (FIX: interleaved [on,vel,on,vel,...])
        pred_view = pred_grid.reshape(pred_grid.shape[0], drum_channels, 2)
        pred_onset = pred_view[:, :, 0]
        pred_velocity = pred_view[:, :, 1]

        target_view = target_grid.reshape(target_grid.shape[0], drum_channels, 2)
        target_onset = target_view[:, :, 0]
        target_velocity = target_view[:, :, 1]

        
        # Extract notes
        pred_notes = []
        target_notes = []
        
        frame_to_sec = 1.0 / self.config.FPS
        
        def denorm_vel(val):
            return np.clip(((val + 1) / 2) * 127, 1, 127)
        
        for drum_idx in range(drum_channels):
            # Predictions - use find_peaks for onset detection
            # [유지] 사용자가 의도한 대로 threshold 주석 처리 유지 (기본값 0.0)
            current_threshold = 0.0
            # if drum_idx == 1:  # Snare - lower threshold for better recall
            #     current_threshold = -0.5
            # elif drum_idx == 3:  # Toms - lower threshold for better recall  
            # #     current_threshold = -0.5
            # elif drum_idx == 4:  # Crash - higher threshold for better precision
            #     current_threshold = 0.5
            # elif drum_idx == 6:
            #     current_threshold = -0.5
                
            pred_peaks, _ = find_peaks(pred_onset[:, drum_idx], height=current_threshold, distance=3)
            for peak_frame in pred_peaks:
                onset_time = peak_frame * frame_to_sec + time_offset
                raw_vel = pred_velocity[peak_frame, drum_idx]
                real_vel = denorm_vel(raw_vel)
                pred_notes.append([onset_time, drum_idx + 1, real_vel])
                
            # Targets
            target_peaks = np.where(target_onset[:, drum_idx] > -0.5)[0]
            for peak_frame in target_peaks:
                onset_time = peak_frame * frame_to_sec + time_offset
                raw_vel = target_velocity[peak_frame, drum_idx]
                real_vel = denorm_vel(raw_vel)
                target_notes.append([onset_time, drum_idx + 1, real_vel])
        
        pred_arr = np.array(pred_notes) if len(pred_notes) > 0 else np.empty((0, 3))
        target_arr = np.array(target_notes) if len(target_notes) > 0 else np.empty((0, 3))
        
        # Prepare mir_eval input formats
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

        # 1. Onset Only Metrics (50ms, No Velocity)
        # [수정] mir_eval 사용하여 중복 카운팅 방지 및 표준 계산 방식 적용
        try:
            if len(t_intervals) == 0:
                o_prec, o_rec, o_f1 = 0.0, 0.0, 0.0
            elif len(p_intervals) == 0:
                o_prec, o_rec, o_f1 = 0.0, 0.0, 0.0
            else:
                o_prec, o_rec, o_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=t_intervals,
                    ref_pitches=t_pitches,
                    est_intervals=p_intervals,
                    est_pitches=p_pitches,
                    onset_tolerance=0.05,
                    pitch_tolerance=0.0,
                    offset_ratio=None
                )
                
        except Exception as e:
            print(f"Warning: Error in onset-only metrics calculation: {e}")
            o_prec, o_rec, o_f1 = 0.0, 0.0, 0.0

        # 2. Onset + Velocity Metrics (50ms + 10% Vel)
        # [수정] 위에서 수정한 calculate_onset_velocity_metrics (mir_eval 기반) 호출
        v_prec, v_rec, v_f1 = self.calculate_onset_velocity_metrics(
            t_intervals, t_pitches, t_vels, p_intervals, p_pitches, p_vels
        )
        
        # 3. Per-Drum Metrics (Onset + Velocity)  [기존 유지]
        per_drum_results = {}
        # ✅ [추가] Per-Drum Metrics (Onset Only)
        per_drum_onset_only_results = {}

        for i, drum_name in enumerate(DRUM_NAMES):
            pitch_id = i + 1
            
            if len(p_pitches) > 0:
                p_mask = p_pitches == pitch_id
                sub_p_intervals = p_intervals[p_mask]
                sub_p_pitches = p_pitches[p_mask]
                sub_p_vels = p_vels[p_mask]
            else:
                sub_p_intervals, sub_p_pitches, sub_p_vels = np.empty((0, 2)), np.array([]), np.array([])
                
            if len(t_pitches) > 0:
                t_mask = t_pitches == pitch_id
                sub_t_intervals = t_intervals[t_mask]
                sub_t_pitches = t_pitches[t_mask]
                sub_t_vels = t_vels[t_mask]
            else:
                sub_t_intervals, sub_t_pitches, sub_t_vels = np.empty((0, 2)), np.array([]), np.array([])
            
            # (a) onset+velocity per-drum [기존 유지]
            d_prec, d_rec, d_f1 = self.calculate_onset_velocity_metrics(
                sub_t_intervals, sub_t_pitches, sub_t_vels,
                sub_p_intervals, sub_p_pitches, sub_p_vels
            )
            per_drum_results[drum_name] = {
                'precision': d_prec,
                'recall': d_rec,
                'f1_score': d_f1
            }

            # (b) onset-only per-drum [추가]
            do_prec, do_rec, do_f1 = self.calculate_onset_only_metrics(
                sub_t_intervals, sub_t_pitches,
                sub_p_intervals, sub_p_pitches
            )
            per_drum_onset_only_results[drum_name] = {
                'precision': do_prec,
                'recall': do_rec,
                'f1_score': do_f1
            }
            
        metrics = {
            'onset_only': {
                'precision': o_prec,
                'recall': o_rec,
                'f1_score': o_f1
            },
            'onset_velocity': {
                'precision': v_prec,
                'recall': v_rec,
                'f1_score': v_f1
            },
            'per_drum': per_drum_results,
            # ✅ [추가] per-drum onset-only 결과도 저장
            'per_drum_onset_only': per_drum_onset_only_results
        }
        
        return metrics

    def print_results(self, results):
        """Print formatted evaluation results matching evaluate_focalLoss.py style"""
        print("\n" + "="*60)
        print(f"🎯 EVALUATION RESULTS - Epoch {results['model_info']['checkpoint_epoch']}")
        print("="*60)
        
        print(f"\n📊 Loss & Summary:")
        print(f"   Average Loss: {results['summary']['avg_loss']:.4f}")
        print(f"   Total Files: {results['summary']['num_files']}")
        print(f"   Total Segments: {results['summary']['num_segments']}")
        
        # Onset Only metrics
        onset_avg = results['summary']['onset_only_average']
        print(f"\n🎼 Note-Level Performance (Onset Only, 50ms):")
        print(f"   F1-Score: {onset_avg['f1_score']:.3f} | P: {onset_avg['precision']:.3f}, R: {onset_avg['recall']:.3f}")
        
        # Onset + Velocity metrics
        velocity_avg = results['summary']['onset_velocity_average']
        print(f"\n🎼 Note-Level Performance (Onset + Velocity, 50ms):")
        print(f"   F1-Score: {velocity_avg['f1_score']:.3f} | P: {velocity_avg['precision']:.3f}, R: {velocity_avg['recall']:.3f}")
        
        print(f"\n🥁 Per-Drum Scores (Onset+Velocity):")
        per_drum_avg = results['summary']['per_drum_average']
        for drum_name in DRUM_NAMES:
            metrics = per_drum_avg[drum_name]
            print(f"   {drum_name:6}: F1 {metrics['f1_score']:.3f} (P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f})")

        # ✅ [추가] Per-Drum Onset Only 출력
        if 'per_drum_onset_only_average' in results['summary']:
            print(f"\n🥁 Per-Drum Scores (Onset Only, 50ms):")
            per_drum_onset_only_avg = results['summary']['per_drum_onset_only_average']
            for drum_name in DRUM_NAMES:
                m = per_drum_onset_only_avg[drum_name]
                print(f"   {drum_name:6}: F1 {m['f1_score']:.3f} (P: {m['precision']:.3f}, R: {m['recall']:.3f})")
        
        final_f1 = velocity_avg['f1_score']
        
        # Performance Assessment
        print(f"\n🎯 Performance Assessment:")
        print(f"   현재 F1: {final_f1:.3f} | 논문 EGMD F1: 0.826 (reference)")
        if final_f1 >= 0.8:
            print("   🔥 EXCELLENT - Near paper-level performance!")
        elif final_f1 >= 0.6:
            print("   ✅ GOOD - Strong drum transcription capability")
        elif final_f1 >= 0.4:
            print("   📈 MODERATE - Learning in progress")
        else:
            print("   ⚠️  BASIC - Early learning stage")

    def save_results(self, results):
        """Save results in JSON format matching evaluate_focalLoss.py style"""
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            epoch = results['model_info']['checkpoint_epoch']
            output_file = os.path.join(self.args.output_dir, f"eval_stitched_ep{epoch}.json")
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")

    @torch.no_grad()
    def evaluate_model(self):
        """Evaluate model using proper stitch-first approach"""
        if self.args.quick:
            print("\n🚀 Starting Quick Evaluation (200 random samples)...")
            eval_dataset = QuickEGMDEvalDataset(num_samples=200, overlap_ratio=0.5)
        else:
            print("\n🎯 Starting Stitch-First Evaluation...")
            eval_dataset = EGMDEvalDataset(overlap_ratio=0.5)
        
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        eval_mode = "Quick" if self.args.quick else "Full"
        print(f"Processing {len(eval_dataset)} segments in {eval_mode} mode...")
        
        # Phase 1: Collect all predictions and targets for each file
        segment_results = []
        all_losses = []
        
        progress_desc = "Quick Eval: Collecting predictions" if self.args.quick else "Phase 1: Collecting predictions"
        progress_bar = tqdm(eval_loader, desc=progress_desc)
        
        for batch_idx, (audio_mert, spec, target_grid) in enumerate(progress_bar):
            # Move to device
            audio_mert = audio_mert.to(self.device)
            spec = spec.to(self.device)
            target_grid = target_grid.to(self.device)
            
            # Get segment info
            segment_info = eval_dataset.files[batch_idx]
            
            # Calculate loss
            loss = self.loss_fn(audio_mert, spec, target_grid, progress=0.5)
            all_losses.append(loss.mean().item())
            
            # Generate predictions
            predictions = self.loss_fn.sample(audio_mert, spec, steps=self.args.eval_steps)
            predictions = predictions[0].cpu().numpy()
            target_grid = target_grid[0].cpu().numpy()
            
            # Store segment result for stitching
            segment_result = {
                'file_id': segment_info['file_id'],
                'start_time': segment_info['start_time'],
                'end_time': segment_info['end_time'],
                'pred_grid': predictions,
                'target_grid': target_grid,
                'loss': loss.mean().item()
            }
            segment_results.append(segment_result)
            
            progress_bar.set_postfix({
                'loss': f'{loss.mean().item():.4f}',
                'file': segment_info['file_id'][:15] + '...' if len(segment_info['file_id']) > 15 else segment_info['file_id']
            })
        
        # Phase 2: Stitch predictions for each file
        phase2_desc = "Quick Eval: Stitching segments" if self.args.quick else "🔄 Phase 2: Stitching overlapping segments"
        print(f"\n{phase2_desc}...")
        file_predictions, file_targets = self.stitch_predictions(segment_results)
        
        # Phase 3: Calculate metrics on stitched files
        phase3_desc = "Quick Eval: Computing metrics" if self.args.quick else "📊 Phase 3: Computing metrics on stitched files"
        print(f"\n{phase3_desc}...")
        
        file_metrics = {}
        for file_id in file_predictions:
            pred_grid = file_predictions[file_id]
            target_grid = file_targets[file_id]
            
            metrics = self.calculate_file_metrics(pred_grid, target_grid)
            file_metrics[file_id] = metrics
        
        # Aggregate metrics across all files
        stats_desc = "Quick Eval: Computing statistics" if self.args.quick else "📈 Computing Overall Statistics"
        print(f"\n{stats_desc}...")
        
        avg_loss = np.mean(all_losses)
        
        # Aggregate onset-only metrics
        onset_precisions = [file_metrics[file_id]['onset_only']['precision'] for file_id in file_metrics]
        onset_recalls = [file_metrics[file_id]['onset_only']['recall'] for file_id in file_metrics]
        onset_f1s = [file_metrics[file_id]['onset_only']['f1_score'] for file_id in file_metrics]
        
        onset_avg = {
            'precision': np.mean(onset_precisions),
            'recall': np.mean(onset_recalls),
            'f1_score': np.mean(onset_f1s)
        }
        
        # Aggregate onset+velocity metrics
        velocity_precisions = [file_metrics[file_id]['onset_velocity']['precision'] for file_id in file_metrics]
        velocity_recalls = [file_metrics[file_id]['onset_velocity']['recall'] for file_id in file_metrics]
        velocity_f1s = [file_metrics[file_id]['onset_velocity']['f1_score'] for file_id in file_metrics]
        
        velocity_avg = {
            'precision': np.mean(velocity_precisions),
            'recall': np.mean(velocity_recalls),
            'f1_score': np.mean(velocity_f1s)
        }
        
        # Aggregate per-drum metrics (Onset+Velocity) [기존 유지]
        per_drum_avg = {}
        for drum_name in DRUM_NAMES:
            drum_precisions = [file_metrics[file_id]['per_drum'][drum_name]['precision'] for file_id in file_metrics]
            drum_recalls = [file_metrics[file_id]['per_drum'][drum_name]['recall'] for file_id in file_metrics]
            drum_f1s = [file_metrics[file_id]['per_drum'][drum_name]['f1_score'] for file_id in file_metrics]
            
            per_drum_avg[drum_name] = {
                'precision': np.mean(drum_precisions),
                'recall': np.mean(drum_recalls),
                'f1_score': np.mean(drum_f1s)
            }

        # ✅ [추가] Aggregate per-drum onset-only metrics
        per_drum_onset_only_avg = {}
        for drum_name in DRUM_NAMES:
            drum_precisions = [file_metrics[file_id]['per_drum_onset_only'][drum_name]['precision'] for file_id in file_metrics]
            drum_recalls = [file_metrics[file_id]['per_drum_onset_only'][drum_name]['recall'] for file_id in file_metrics]
            drum_f1s = [file_metrics[file_id]['per_drum_onset_only'][drum_name]['f1_score'] for file_id in file_metrics]
            per_drum_onset_only_avg[drum_name] = {
                'precision': np.mean(drum_precisions),
                'recall': np.mean(drum_recalls),
                'f1_score': np.mean(drum_f1s)
            }
        
        # Print results
        eval_mode = "Quick" if self.args.quick else "Stitch-First"
        result_title = f"🚀 {eval_mode} Evaluation Results" if self.args.quick else f"🎯 {eval_mode} Evaluation Results"
        print(f"\n{result_title}:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Total Files: {len(file_metrics)}")
        print(f"   Total Segments Processed: {len(segment_results)}")
        
        print(f"\n📁 Onset-Only Metrics (mean across {len(file_metrics)} files):")
        print(f"   Overall: P={onset_avg['precision']:.3f} R={onset_avg['recall']:.3f} F1={onset_avg['f1_score']:.3f}")
        
        print(f"\n📁 Onset+Velocity Metrics (mean across {len(file_metrics)} files):")
        print(f"   Overall: P={velocity_avg['precision']:.3f} R={velocity_avg['recall']:.3f} F1={velocity_avg['f1_score']:.3f}")
        
        print(f"\n🥁 Per-Drum Onset+Velocity:")
        for drum_name in DRUM_NAMES:
            m = per_drum_avg[drum_name]
            print(f"   {drum_name:8}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1_score']:.3f}")

        # ✅ [추가] per-drum onset-only 출력(요청사항)
        print(f"\n🥁 Per-Drum Onset Only (50ms):")
        for drum_name in DRUM_NAMES:
            m = per_drum_onset_only_avg[drum_name]
            print(f"   {drum_name:8}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1_score']:.3f}")
        
        # Save detailed results
        results = {
            'model_info': {
                'checkpoint_epoch': self.epoch,
                'training_loss': self.train_loss,
                'evaluation_mode': 'quick' if self.args.quick else 'full'
            },
            'summary': {
                'avg_loss': avg_loss,
                'num_files': len(file_metrics),
                'num_segments': len(segment_results),
                'onset_only_average': onset_avg,
                'onset_velocity_average': velocity_avg,
                'per_drum_average': per_drum_avg,
                # ✅ [추가] summary에도 per-drum onset-only 평균 추가
                'per_drum_onset_only_average': per_drum_onset_only_avg
            },
            'per_file_metrics': file_metrics,
            'segment_results': segment_results
        }
        
        # Set output filename based on mode
        if self.args.quick:
            results_filename = f'quick_eval_ep{self.epoch}.json'
            completion_msg = "✅ Quick Evaluation completed!"
        else:
            results_filename = f'full_eval_ep{self.epoch}.json'
            completion_msg = "✅ Stitch-First Evaluation completed!"
            
        results_path = os.path.join(self.args.output_dir, results_filename)
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print and save results
        self.print_results(results)
        
        print(f"\n{completion_msg}")
        
        return results

def parse_args():
    parser = argparse.ArgumentParser(description='N2N-Flow2 Proper Evaluation with Stitching')
    parser.add_argument('--ckpt_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='./eval_results', help='Output directory')
    parser.add_argument('--eval_steps', type=int, default=5, help='Number of sampling steps')
    parser.add_argument('--quick', action='store_true', help='Run quick evaluation on 200 random samples')
    return parser.parse_args()

def main():
    args = parse_args()
    evaluator = ProperEvaluator(args)
    results = evaluator.evaluate_model()
    return results

if __name__ == "__main__":
    main()
