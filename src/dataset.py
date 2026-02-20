# src/dataset.py
import os
import glob
import torch
import torchaudio
import torch.nn.functional as F
import pretty_midi
import numpy as np
import json
import math
from torch.utils.data import Dataset, RandomSampler, WeightedRandomSampler
from tqdm import tqdm
from .config import Config

# E-GMD MIDI Note -> 7 Classes Mapping
DRUM_MAPPING = {
    # Kick
    35: 0, 36: 0, 
    # Snare
    38: 1, 40: 1, 37: 1, 
    # Hi-hat (Closed, Pedal, Open)
    42: 2, 44: 2, 46: 2, 
    # Toms (Low, Mid, High)
    41: 3, 43: 3, 45: 3, 47: 3, 48: 3, 50: 3, 
    # Crash
    49: 4, 57: 4, 55: 4, 52: 4, 
    # Ride
    51: 5, 59: 5, 53: 5, 
    # Bell
    56: 6, 54: 6 
}



def get_egmd_train_sampler(train_dataset, oversample=False):
    """Return sampler for EGMD training.

    Args:
        train_dataset: EGMDTrainDataset instance.
        oversample: If True, use WeightedRandomSampler with replacement.
                    If False, use RandomSampler without replacement.
    """
    if oversample:
        if getattr(train_dataset, "sample_weights", None) is None:
            raise ValueError("sample_weights is required when oversample=True")
        return WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset.sample_weights),
            replacement=True
        )

    return RandomSampler(train_dataset, replacement=False)


class EGMDTrainDataset(Dataset):
    """EGMD Training Dataset - only uses train sessions"""
    def __init__(self, compute_sample_weights=True):
        self.config = Config()
        self.files = []
        # Per-sample weights for WeightedRandomSampler (computed after files are loaded)
        self.sample_weights = None
        
        # Spectrogram Transform (44.1kHz)
        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.AUDIO_SR,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            normalized=True
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # Resampler for MERT (44.1k -> 24k)
        self.resampler_to_mert = torchaudio.transforms.Resample(
            self.config.AUDIO_SR, self.config.MERT_SR
        )
        self._resampler_cache = {}
        
        # Load training data only
        self._load_train_data()
        # Compute per-file/sample weights for oversampling rare-hit files
        if compute_sample_weights:
            self.sample_weights = self._compute_sample_weights()
        print(f"Found {len(self.files)} training pairs.")


    def _cache_path(self):
        """Where to store precomputed per-file sampling weights."""
        cache_dir = os.path.join(self.config.DATA_ROOT, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        # bump v1 -> v2 if you change the weight formula / mapping
        return os.path.join(cache_dir, "egmd_train_sampling_weights_v1.json")

    def _load_cached_weights(self):
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "weights_by_midi" not in data:
                return None
            return data
        except Exception as e:
            print(f"[WARN] Failed to load sampling-weight cache: {e}")
            return None

    def _save_cached_weights(self, payload):
        path = self._cache_path()
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"[WARN] Failed to save sampling-weight cache: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass

    def _extract_midi_hit_counts(self, midi_path):
        """Return per-class hit counts for a MIDI file."""
        counts = [0] * self.config.DRUM_CHANNELS
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            for instrument in pm.instruments:
                if instrument.is_drum:
                    for note in instrument.notes:
                        if note.pitch in DRUM_MAPPING:
                            counts[DRUM_MAPPING[note.pitch]] += 1
        except Exception as e:
            print(f"[WARN] MIDI parse failed ({midi_path}): {e}")
        return counts

    def _compute_sample_weights(self):
        """Compute per-(audio,midi) pair sampling weights for WeightedRandomSampler.

        Goal: oversample files that contain more 'rare' hits (TT/CY/RD/BE).
        We cache results to disk so --resume doesn't re-scan MIDIs every run.
        """
        multipliers = {
            0: float(getattr(self.config, 'OVERSAMPLE_KICK', 1.0)),
            1: float(getattr(self.config, 'OVERSAMPLE_SNARE', 1.0)),
            2: float(getattr(self.config, 'OVERSAMPLE_HH', 1.0)),
            3: float(getattr(self.config, 'OVERSAMPLE_TOMS', 2.0)),
            4: float(getattr(self.config, 'OVERSAMPLE_CRASH', 3.0)),
            5: float(getattr(self.config, 'OVERSAMPLE_RIDE', 2.0)),
            6: float(getattr(self.config, 'OVERSAMPLE_BELL', 4.0)),
        }  # KD, SN, HH, TT, CY, RD, BE
        count_strength = {
            0: float(getattr(self.config, 'COUNT_STRENGTH_KICK', 0.3)),
            1: float(getattr(self.config, 'COUNT_STRENGTH_SNARE', 0.3)),
            2: float(getattr(self.config, 'COUNT_STRENGTH_HH', 0.3)),
            3: float(getattr(self.config, 'COUNT_STRENGTH_TOMS', 0.6)),
            4: float(getattr(self.config, 'COUNT_STRENGTH_CRASH', 0.8)),
            5: float(getattr(self.config, 'COUNT_STRENGTH_RIDE', 0.7)),
            6: float(getattr(self.config, 'COUNT_STRENGTH_BELL', 0.9)),
        }

        cached = self._load_cached_weights()
        weights_by_midi = {}
        counts_by_midi = {}
        if cached is not None:
            weights_by_midi = cached.get("weights_by_midi", {}) or {}
            counts_by_midi = cached.get("counts_by_midi", {}) or {}

        updated = False
        weights = []

        for _, midi_path in tqdm(self.files, desc="Computing sampling weights", unit="file"):
            key = os.path.relpath(midi_path, self.config.DATA_ROOT)

            if key in weights_by_midi:
                weights.append(float(weights_by_midi[key]))
                continue

            counts = self._extract_midi_hit_counts(midi_path)
            w = 1.0

            for c, mult in multipliers.items():
                cnt = counts[c]
                if cnt > 0:
                    w *= mult
                    denom = math.log1p(50.0)
                    w *= math.exp(count_strength[c] * (math.log1p(float(cnt)) / denom))

            w = float(max(0.1, min(w, 20.0)))

            weights.append(w)
            weights_by_midi[key] = w
            counts_by_midi[key] = counts
            updated = True

        if updated or cached is None:
            payload = {
                "version": 1,
                "data_root": self.config.DATA_ROOT,
                "num_files": len(self.files),
                "weights_by_midi": weights_by_midi,
                "counts_by_midi": counts_by_midi,
                "note": "Auto-generated by EGMDTrainDataset for WeightedRandomSampler."
            }
            self._save_cached_weights(payload)
            print(f"[INFO] Saved sampling-weight cache: {self._cache_path()}")

        return torch.DoubleTensor(weights)

    def _load_train_data(self):
        """Load only training data from EGMD (excluding eval and validation sessions)"""
        print(f"Scanning training data from {self.config.DATA_ROOT}...")
        search_path = os.path.join(self.config.DATA_ROOT, "drummer*")
        drummer_dirs = glob.glob(search_path)
        
        for d_dir in drummer_dirs:
            # EGMD structure: train, validation, eval_session
            # We want ONLY train data
            train_dir = os.path.join(d_dir, "train")
            
            if os.path.exists(train_dir):
                # Load from train directory
                audio_files = glob.glob(os.path.join(train_dir, "**", "*.wav"), recursive=True)
            else:
                # Fallback: scan all files but exclude eval_session and validation
                audio_files = glob.glob(os.path.join(d_dir, "**", "*.wav"), recursive=True)
                
            for aud_path in audio_files:
                # Skip eval_session and validation files
                if "eval_session" in aud_path or "validation" in aud_path:
                    continue
                    
                # Find corresponding MIDI file
                mid_path = aud_path.replace(".wav", ".mid")
                if not os.path.exists(mid_path):
                    mid_path = aud_path.replace(".wav", ".midi")
                
                if os.path.exists(mid_path):
                    self.files.append((aud_path, mid_path))
    
    def _get_resampler(self, orig_sr, target_sr):
        """Cached resampler to prevent repeated creation"""
        key = (orig_sr, target_sr)
        if key not in self._resampler_cache:
            self._resampler_cache[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[key]

    def __len__(self):
        return len(self.files)

    def midi_to_grid(self, midi_path, duration):
        """MIDI file to (Time, Channel, [Onset, Vel]) grid conversion"""
        n_frames = int(duration * self.config.FPS)
        grid = np.zeros((n_frames, self.config.DRUM_CHANNELS, 2), dtype=np.float32)
        
        # Initialize: -1 (Silence)
        grid[:, :, 1] = -1.0 
        grid[:, :, 0] = -1.0 
        
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            for instrument in pm.instruments:
                if instrument.is_drum:
                    for note in instrument.notes:
                        if note.pitch in DRUM_MAPPING:
                            idx = DRUM_MAPPING[note.pitch]
                            frame = int(note.start * self.config.FPS)
                            
                            if frame < n_frames:
                                # Onset: 1.0
                                grid[frame, idx, 0] = 1.0
                                # Velocity: Normalize 0~127 -> -1 ~ 1
                                norm_vel = (note.velocity / 127.0) * 2 - 1
                                grid[frame, idx, 1] = norm_vel
        except Exception as e:
            print(f"Error parsing MIDI {midi_path}: {e}")
            
        return grid.reshape(n_frames, -1)

    def __getitem__(self, idx):
        aud_path, mid_path = self.files[idx]
        
        # 1. Load Audio
        wav, sr = torchaudio.load(aud_path)
        
        # 2. Resample for Spectrogram (44.1k)
        if sr != self.config.AUDIO_SR:
            resampler = self._get_resampler(sr, self.config.AUDIO_SR)
            wav_spec_in = resampler(wav)
        else:
            wav_spec_in = wav

        # 3. Resample for MERT (24k)
        if sr == self.config.AUDIO_SR:
            wav_mert_in = self.resampler_to_mert(wav)
        elif sr != self.config.MERT_SR:
            resampler_mert = self._get_resampler(sr, self.config.MERT_SR)
            wav_mert_in = resampler_mert(wav)
        else:
            wav_mert_in = wav
            
        # 4. Random Crop (Training)
        full_len_spec = wav_spec_in.shape[1]
        seg_len_spec = int(self.config.SEGMENT_SEC * self.config.AUDIO_SR)
        
        ratio_mert = self.config.MERT_SR / self.config.AUDIO_SR
        seg_len_mert = int(self.config.SEGMENT_SEC * self.config.MERT_SR)
        
        if full_len_spec > seg_len_spec:
            # Random Start Point
            # randint upper-bound is exclusive, so +1 to include the last valid crop start.
            start_spec = np.random.randint(0, full_len_spec - seg_len_spec + 1)
            start_mert = int(start_spec * ratio_mert)
            
            wav_crop_spec = wav_spec_in[:, start_spec : start_spec + seg_len_spec]
            wav_crop_mert = wav_mert_in[:, start_mert : start_mert + seg_len_mert]

            # NOTE:
            # Due to resampling + integer rounding, MERT length can be a few samples shorter
            # than the exact ratio-based expectation near the tail. Ensure fixed segment length
            # so DataLoader collation remains stable.
            if wav_crop_mert.shape[1] < seg_len_mert:
                wav_crop_mert = torch.nn.functional.pad(
                    wav_crop_mert, (0, seg_len_mert - wav_crop_mert.shape[1])
                )
            elif wav_crop_mert.shape[1] > seg_len_mert:
                wav_crop_mert = wav_crop_mert[:, :seg_len_mert]
            
            start_sec = start_spec / self.config.AUDIO_SR
        else:
            # Padding if too short
            pad_len_spec = seg_len_spec - full_len_spec
            if pad_len_spec > 0:
                wav_crop_spec = torch.nn.functional.pad(wav_spec_in, (0, pad_len_spec))
            else:
                wav_crop_spec = wav_spec_in[:, :seg_len_spec]
                
            pad_len_mert = seg_len_mert - wav_mert_in.shape[1]
            if pad_len_mert > 0:
                wav_crop_mert = torch.nn.functional.pad(wav_mert_in, (0, pad_len_mert))
            else:
                wav_crop_mert = wav_mert_in[:, :seg_len_mert]
            
            start_sec = 0
            
        # 5. Generate Spectrogram
        spec = self.spec_transform(wav_crop_spec[0])
        spec = self.db_transform(spec)
        spec = spec.transpose(0, 1) # (Time, Freq)
        
        # 6. Load MIDI Grid
        total_duration = full_len_spec / self.config.AUDIO_SR
        full_grid = self.midi_to_grid(mid_path, total_duration)
        
        start_frame = int(start_sec * self.config.FPS)
        n_frame_seg = int(self.config.SEGMENT_SEC * self.config.FPS)
        
        grid_crop = full_grid[start_frame : start_frame + n_frame_seg]
        
        # Grid Padding if needed
        if grid_crop.shape[0] < n_frame_seg:
            pad_len = n_frame_seg - grid_crop.shape[0]
            padding = np.ones((pad_len, self.config.DRUM_CHANNELS * 2)) * -1
            grid_crop = np.vstack([grid_crop, padding])
        
        # Align grid with spectrogram length
        target_len = spec.shape[0]
        grid_tensor = torch.from_numpy(grid_crop).float()
        
        if grid_tensor.shape[0] != target_len:
            grid_tensor = grid_tensor.unsqueeze(0).permute(0, 2, 1)
            grid_tensor = F.interpolate(grid_tensor, size=target_len, mode='nearest')
            grid_tensor = grid_tensor.permute(0, 2, 1).squeeze(0)
            
        return wav_crop_mert[0], spec, grid_tensor


class EGMDEvalDataset(Dataset):
    """EGMD Evaluation Dataset - uses sliding windows for 5-second segments"""
    def __init__(self, overlap_ratio=0.5, limit=None):
        self.config = Config()
        self.files = []  # Will store (file_path, segment_info) tuples
        self.overlap_ratio = overlap_ratio
        self.segment_len = 5.0  # 5 seconds per segment (MERT window)
        
        # Spectrogram Transform (44.1kHz)
        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.AUDIO_SR,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            normalized=True
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # Resampler for MERT (44.1k -> 24k)
        self.resampler_to_mert = torchaudio.transforms.Resample(
            self.config.AUDIO_SR, self.config.MERT_SR
        )
        self._resampler_cache = {}
        
        # Audio file cache to avoid repeated loading/resampling
        self._audio_cache = {}  # {file_path: {'spec': tensor, 'mert': tensor, 'sr': int}}
        self._cache_lock = {}   # Thread safety for caching
        
        # Load evaluation data and create sliding windows
        self._load_eval_data_with_windows(limit)
    
    def _load_eval_data_with_windows(self, limit=None):
        """Load eval_session data and create sliding window segments"""
        print(f"Scanning eval_session data from {self.config.DATA_ROOT}...")
        search_path = os.path.join(self.config.DATA_ROOT, "drummer*")
        drummer_dirs = glob.glob(search_path)
        
        total_files = 0
        
        for d_dir in drummer_dirs:
            # Some drummers have eval_session, some don't - handle gracefully
            eval_session_paths = [
                os.path.join(d_dir, "eval_session"),
                os.path.join(d_dir, "evaluation"),  # Alternative naming
            ]
            
            eval_dir_found = False
            for eval_path in eval_session_paths:
                if os.path.exists(eval_path):
                    audio_files = glob.glob(os.path.join(eval_path, "**", "*.wav"), recursive=True)
                    
                    for aud_path in audio_files:
                        # Find corresponding MIDI file
                        mid_path = aud_path.replace(".wav", ".mid")
                        if not os.path.exists(mid_path):
                            mid_path = aud_path.replace(".wav", ".midi")
                        
                        if os.path.exists(mid_path):
                            # Create sliding window segments for this file
                            segments = self._create_segments(aud_path, mid_path)
                            self.files.extend(segments)
                            total_files += 1
                    
                    eval_dir_found = True
                    break
            
            # Fallback: look for eval_session in file names if no eval directory
            if not eval_dir_found:
                audio_files = glob.glob(os.path.join(d_dir, "**", "*.wav"), recursive=True)
                for aud_path in audio_files:
                    if "eval_session" in aud_path or "evaluation" in aud_path:
                        mid_path = aud_path.replace(".wav", ".mid")
                        if not os.path.exists(mid_path):
                            mid_path = aud_path.replace(".wav", ".midi")
                        
                        if os.path.exists(mid_path):
                            # Create sliding window segments for this file
                            segments = self._create_segments(aud_path, mid_path)
                            self.files.extend(segments)
                            total_files += 1
        
        print(f"Processed {total_files} evaluation files into {len(self.files)} segments")
        
        # === [핵심 수정 부분] 캐싱하기 전에 리스트 먼저 자르기 ===
        if limit is not None and len(self.files) > limit:
            import random
            random.seed(43)
            random.shuffle(self.files)  # 랜덤 섞기
            self.files = self.files[:limit]  # 리스트 자르기
            print(f"⚡ Optimized Loading: Limited to {len(self.files)} segments.")
            
        print(f"Pre-loading and caching audio files for faster evaluation...")
        self._preload_audio_cache()
    
    def _preload_audio_cache(self):
        """Pre-load and cache all audio files to avoid repeated I/O"""
        unique_audio_paths = set(segment['audio_path'] for segment in self.files)
        
        for audio_path in tqdm(unique_audio_paths, desc="Caching audio files"):
            if audio_path not in self._audio_cache:
                try:
                    wav, sr = torchaudio.load(audio_path)
                    
                    # Resample for Spectrogram (44.1k)
                    if sr != self.config.AUDIO_SR:
                        resampler = self._get_resampler(sr, self.config.AUDIO_SR)
                        wav_spec_full = resampler(wav)
                    else:
                        wav_spec_full = wav

                    # Resample for MERT (24k)
                    if sr == self.config.AUDIO_SR:
                        wav_mert_full = self.resampler_to_mert(wav)
                    elif sr != self.config.MERT_SR:
                        resampler_mert = self._get_resampler(sr, self.config.MERT_SR)
                        wav_mert_full = resampler_mert(wav)
                    else:
                        wav_mert_full = wav
                    
                    # Cache the resampled audio
                    self._audio_cache[audio_path] = {
                        'spec': wav_spec_full,
                        'mert': wav_mert_full,
                        'original_sr': sr
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to cache {audio_path}: {e}")
        
        print(f"Cached {len(self._audio_cache)} audio files successfully!")
    def _create_segments(self, aud_path, mid_path):
        """Create sliding window segments from a single audio/MIDI file pair"""
        # Get audio duration
        try:
            info = torchaudio.info(aud_path)
            duration = info.num_frames / info.sample_rate
        except:
            print(f"Warning: Could not get duration for {aud_path}")
            return []
        
        segments = []
        
        # Create unique file ID using full relative path to prevent collisions
        # Remove common root to get clean relative path
        rel_path = os.path.relpath(aud_path, self.config.DATA_ROOT)
        unique_file_id = rel_path.replace(os.sep, '_').replace('.wav', '')
        
        if duration <= self.segment_len:
            # If file is shorter than segment length, use the whole file
            segments.append({
                'audio_path': aud_path,
                'midi_path': mid_path,
                'start_time': 0.0,
                'end_time': duration,
                'file_id': unique_file_id,
                'segment_idx': 0
            })
        else:
            # Create sliding windows
            hop_size = self.segment_len * (1 - self.overlap_ratio)  # 50% overlap = 2.5s hop
            
            segment_idx = 0
            start_time = 0.0
            
            while start_time < duration:
                end_time = min(start_time + self.segment_len, duration)
                
                # Include all segments, even short ones (no 2-second minimum)
                segments.append({
                    'audio_path': aud_path,
                    'midi_path': mid_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'file_id': unique_file_id,
                    'segment_idx': segment_idx
                })
                
                segment_idx += 1
                start_time += hop_size
                
                # If this would be the last segment and it starts very close to the end, still include it
                if start_time >= duration:
                    break
        
        return segments
    
    def _get_resampler(self, orig_sr, target_sr):
        """Cached resampler to prevent repeated creation"""
        key = (orig_sr, target_sr)
        if key not in self._resampler_cache:
            self._resampler_cache[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resampler_cache[key]

    def __len__(self):
        return len(self.files)

    def midi_to_grid(self, midi_path, duration):
        """MIDI file to (Time, Channel, [Onset, Vel]) grid conversion"""
        n_frames = int(duration * self.config.FPS)
        grid = np.zeros((n_frames, self.config.DRUM_CHANNELS, 2), dtype=np.float32)
        
        # Initialize: -1 (Silence)
        grid[:, :, 1] = -1.0 
        grid[:, :, 0] = -1.0 
        
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            for instrument in pm.instruments:
                if instrument.is_drum:
                    for note in instrument.notes:
                        if note.pitch in DRUM_MAPPING:
                            idx = DRUM_MAPPING[note.pitch]
                            frame = int(note.start * self.config.FPS)
                            
                            if frame < n_frames:
                                # Onset: 1.0
                                grid[frame, idx, 0] = 1.0
                                # Velocity: Normalize 0~127 -> -1 ~ 1
                                norm_vel = (note.velocity / 127.0) * 2 - 1
                                grid[frame, idx, 1] = norm_vel
        except Exception as e:
            print(f"Error parsing MIDI {midi_path}: {e}")
            
        return grid.reshape(n_frames, -1)

    def __getitem__(self, idx):
        segment_info = self.files[idx]
        aud_path = segment_info['audio_path']
        mid_path = segment_info['midi_path']
        start_time = segment_info['start_time']
        end_time = segment_info['end_time']
        
        # 1. Get cached audio data (avoid repeated loading/resampling)
        if aud_path in self._audio_cache:
            cached_data = self._audio_cache[aud_path]
            wav_spec_full = cached_data['spec']
            wav_mert_full = cached_data['mert']
            target_sr_spec = self.config.AUDIO_SR
            target_sr_mert = self.config.MERT_SR
        else:
            # Fallback: direct load if not cached (shouldn't happen with pre-caching)
            print(f"Warning: {aud_path} not in cache, loading directly")
            wav, sr = torchaudio.load(aud_path)
            
            # Resample for Spectrogram (44.1k)
            if sr != self.config.AUDIO_SR:
                resampler = self._get_resampler(sr, self.config.AUDIO_SR)
                wav_spec_full = resampler(wav)
                target_sr_spec = self.config.AUDIO_SR
            else:
                wav_spec_full = wav
                target_sr_spec = sr

            # Resample for MERT (24k)
            if sr == self.config.AUDIO_SR:
                wav_mert_full = self.resampler_to_mert(wav)
                target_sr_mert = self.config.MERT_SR
            elif sr != self.config.MERT_SR:
                resampler_mert = self._get_resampler(sr, self.config.MERT_SR)
                wav_mert_full = resampler_mert(wav)
                target_sr_mert = self.config.MERT_SR
            else:
                wav_mert_full = wav
                target_sr_mert = sr
        
        # 2. Extract segments from cached resampled audio
        start_frame_spec = int(start_time * target_sr_spec)
        end_frame_spec = int(end_time * target_sr_spec)
        end_frame_spec = min(end_frame_spec, wav_spec_full.shape[1])
        
        start_frame_mert = int(start_time * target_sr_mert)
        end_frame_mert = int(end_time * target_sr_mert)
        end_frame_mert = min(end_frame_mert, wav_mert_full.shape[1])
        
        # Extract segments
        wav_spec_in = wav_spec_full[:, start_frame_spec:end_frame_spec]
        wav_mert_in = wav_mert_full[:, start_frame_mert:end_frame_mert]
            
        # 4. Generate Spectrogram for the segment
        spec = self.spec_transform(wav_spec_in[0])
        spec = self.db_transform(spec)
        spec = spec.transpose(0, 1) # (Time, Freq)
        
        # 5. Load MIDI Grid for the segment duration
        segment_duration = end_time - start_time
        segment_grid = self.midi_to_grid_segment(mid_path, start_time, segment_duration)
        
        # Align grid with spectrogram length
        target_len = spec.shape[0]
        grid_tensor = torch.from_numpy(segment_grid).float()
        
        if grid_tensor.shape[0] != target_len:
            grid_tensor = grid_tensor.unsqueeze(0).permute(0, 2, 1)
            grid_tensor = F.interpolate(grid_tensor, size=target_len, mode='nearest')
            grid_tensor = grid_tensor.permute(0, 2, 1).squeeze(0)
            
        return wav_mert_in[0], spec, grid_tensor
    
    def midi_to_grid_segment(self, midi_path, start_time, duration):
        """MIDI file to grid conversion for a specific time segment"""
        n_frames = int(duration * self.config.FPS)
        grid = np.zeros((n_frames, self.config.DRUM_CHANNELS, 2), dtype=np.float32)
        
        # Initialize: -1 (Silence)
        grid[:, :, 1] = -1.0 
        grid[:, :, 0] = -1.0 
        
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            for instrument in pm.instruments:
                if instrument.is_drum:
                    for note in instrument.notes:
                        # Only process notes that fall within our time segment
                        if (note.start >= start_time and 
                            note.start < start_time + duration and
                            note.pitch in DRUM_MAPPING):
                            
                            idx = DRUM_MAPPING[note.pitch]
                            # Adjust frame index relative to segment start
                            frame = int((note.start - start_time) * self.config.FPS)
                            
                            if 0 <= frame < n_frames:
                                # Onset: 1.0
                                grid[frame, idx, 0] = 1.0
                                # Velocity: Normalize 0~127 -> -1 ~ 1
                                norm_vel = (note.velocity / 127.0) * 2 - 1
                                grid[frame, idx, 1] = norm_vel
        except Exception as e:
            print(f"Error parsing MIDI {midi_path}: {e}")
            
        return grid.reshape(n_frames, -1)
