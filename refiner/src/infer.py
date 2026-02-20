import argparse
import os

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import find_peaks
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm import tqdm

from src.config import Config as FMConfig
from src.model import AnnealedPseudoHuberLoss, FlowMatchingTransformer

from .config import RefinerTrainConfig
from .model import DrumRefiner, RefinerConfig, apply_edits

DRUM_MAPPING = [36, 38, 42, 47, 49, 51, 56]


class RefinerInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fm_cfg = FMConfig()
        self.ref_cfg = RefinerTrainConfig()

        # Refiner is trained with 20s windows.
        self.fm_cfg.SEGMENT_SEC = self.ref_cfg.segment_sec

        self.fm_model = FlowMatchingTransformer(self.fm_cfg).to(self.device)
        self.loss_fn = AnnealedPseudoHuberLoss(self.fm_model, self.fm_cfg).to(self.device)
        self.refiner = DrumRefiner(
            RefinerConfig(drum_channels=self.fm_cfg.DRUM_CHANNELS), cond_dim=self.fm_cfg.N_MELS
        ).to(self.device)

        self._load_fm_checkpoint(args.fm_ckpt)
        self._load_refiner_checkpoint(args.refiner_ckpt)

        self.fm_model.eval()
        self.refiner.eval()
        self._init_feature_extractors()

    def _load_fm_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.fm_model.load_state_dict(state_dict, strict=True)

    def _load_refiner_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.refiner.load_state_dict(checkpoint["model"], strict=True)

    def _init_feature_extractors(self):
        self.mel_transform = MelSpectrogram(
            sample_rate=self.fm_cfg.AUDIO_SR,
            n_fft=self.fm_cfg.N_FFT,
            hop_length=self.fm_cfg.HOP_LENGTH,
            n_mels=self.fm_cfg.N_MELS,
            normalized=True,
        ).to(self.device)
        self.db_transform = AmplitudeToDB().to(self.device)
        self.resampler_to_44k = {}
        self.resampler_to_24k = {}

    def _get_features(self, waveform_segment, sr):
        if sr != self.fm_cfg.AUDIO_SR:
            if sr not in self.resampler_to_44k:
                self.resampler_to_44k[sr] = torchaudio.transforms.Resample(sr, self.fm_cfg.AUDIO_SR).to(self.device)
            waveform_mel = self.resampler_to_44k[sr](waveform_segment).to(self.device)
        else:
            waveform_mel = waveform_segment.to(self.device)

        melspec = self.mel_transform(waveform_mel)
        melspec = self.db_transform(melspec)
        melspec = melspec.transpose(1, 2)

        if sr != self.fm_cfg.MERT_SR:
            if sr not in self.resampler_to_24k:
                self.resampler_to_24k[sr] = torchaudio.transforms.Resample(sr, self.fm_cfg.MERT_SR).to(self.device)
            waveform_mert = self.resampler_to_24k[sr](waveform_segment.to(self.device))
        else:
            waveform_mert = waveform_segment.to(self.device)

        waveform_mert = waveform_mert.squeeze(0).unsqueeze(0)
        return waveform_mert, melspec

    @torch.no_grad()
    def _run_fm_refiner_on_segment(self, waveform_mert, spec):
        fm_logits = self.loss_fn.sample(waveform_mert, spec, steps=self.args.fm_steps)
        out = self.refiner(fm_logits, cond_feats=spec)
        out["gate"] = torch.clamp(out["gate"] * self.args.refiner_strength, 0.0, 1.0)
        refined = apply_edits(fm_logits, out["edit_logits"], out["vel_residual"], out["gate"])
        return refined[0].detach().cpu().numpy()

    def _stitch_predictions(self, segment_results, total_duration):
        grid_length = int(total_duration * self.fm_cfg.FPS) + 1
        drum_channels = self.fm_cfg.DRUM_CHANNELS

        stitched_pred = np.full((grid_length, drum_channels * 2), -1.0, dtype=np.float32)
        overlap_count = np.zeros(grid_length, dtype=int)

        segment_results.sort(key=lambda x: x["start_time"])
        for segment in segment_results:
            start_frame = int(segment["start_time"] * self.fm_cfg.FPS)
            pred_grid = segment["pred_grid"]
            segment_length = min(pred_grid.shape[0], grid_length - start_frame)

            if segment_length <= 0:
                continue

            for frame_idx in range(segment_length):
                global_frame = start_frame + frame_idx
                if overlap_count[global_frame] == 0:
                    stitched_pred[global_frame] = pred_grid[frame_idx]
                else:
                    alpha = 1.0 / (overlap_count[global_frame] + 1)
                    stitched_pred[global_frame] = (1 - alpha) * stitched_pred[global_frame] + alpha * pred_grid[frame_idx]
                overlap_count[global_frame] += 1

        return stitched_pred

    def _postprocess_predictions(self, predictions):
        drum_channels = self.fm_cfg.DRUM_CHANNELS
        pred_view = predictions.reshape(predictions.shape[0], drum_channels, 2)
        onset_pred = pred_view[:, :, 0]
        velocity_pred = pred_view[:, :, 1]
        velocity_norm = np.clip(((velocity_pred + 1) / 2) * 127, 1, 127)

        drum_events = []
        for drum_idx in range(drum_channels):
            peaks, _ = find_peaks(
                onset_pred[:, drum_idx],
                height=self.args.threshold,
                distance=int(0.05 * self.fm_cfg.FPS),
            )
            for peak in peaks:
                drum_events.append(
                    {
                        "time": peak / self.fm_cfg.FPS,
                        "drum": drum_idx,
                        "midi_note": DRUM_MAPPING[drum_idx],
                        "velocity": int(velocity_norm[peak, drum_idx]),
                        "onset_score": float(onset_pred[peak, drum_idx]),
                    }
                )

        return sorted(drum_events, key=lambda x: x["time"])

    def _create_midi_file(self, drum_events, output_path):
        pm = pretty_midi.PrettyMIDI()
        drum_program = pretty_midi.instrument_name_to_program("Synth Drum")
        drums = pretty_midi.Instrument(program=drum_program, is_drum=True)

        for event in drum_events:
            drums.notes.append(
                pretty_midi.Note(
                    velocity=event["velocity"],
                    pitch=event["midi_note"],
                    start=event["time"],
                    end=event["time"] + 0.1,
                )
            )

        pm.instruments.append(drums)
        pm.write(output_path)

    def process_audio_file(self, input_path, output_path):
        waveform, sr = torchaudio.load(input_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        total_duration = waveform.shape[1] / sr
        segment_duration = self.ref_cfg.segment_sec
        segment_samples = int(segment_duration * sr)
        hop_duration = segment_duration * (1 - self.args.overlap_ratio)

        num_segments = int(np.ceil((total_duration - segment_duration) / hop_duration)) + 1
        if total_duration <= segment_duration:
            num_segments = 1

        segment_results = []
        for seg_idx in tqdm(range(num_segments), desc=f"Refiner infer: {os.path.basename(input_path)}"):
            start_time = seg_idx * hop_duration
            start_sample = int(start_time * sr)
            end_sample = min(start_sample + segment_samples, waveform.shape[1])

            segment = waveform[:, start_sample:end_sample]
            if segment.shape[1] < segment_samples:
                segment = F.pad(segment, (0, segment_samples - segment.shape[1]))

            waveform_mert, melspec = self._get_features(segment, sr)
            refined = self._run_fm_refiner_on_segment(waveform_mert, melspec)
            segment_results.append({"start_time": start_time, "pred_grid": refined})

        stitched = self._stitch_predictions(segment_results, total_duration)
        events = self._postprocess_predictions(stitched)
        self._create_midi_file(events, output_path)
        print(f"💾 MIDI saved: {output_path} ({len(events)} notes)")


def parse_args():
    p = argparse.ArgumentParser(description="FM+Refiner drum inference")
    p.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    p.add_argument("--output", type=str, required=True, help="Output directory for MIDI files")

    p.add_argument("--fm_ckpt", type=str, required=True, help="Flow-matching checkpoint path")
    p.add_argument("--fm_steps", type=int, default=3, help="Flow-matching sampling steps")
    p.add_argument("--refiner_ckpt", type=str, required=True, help="Refiner checkpoint path")

    p.add_argument("--threshold", type=float, default=0.0, help="Onset threshold for peak picking")
    p.add_argument("--overlap_ratio", type=float, default=0.5, help="Overlap ratio for 20s stitching")
    p.add_argument("--refiner_strength", type=float, default=1.0, help="Scale factor for refiner gate")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    inferencer = RefinerInference(args)

    if os.path.isfile(args.input):
        name = os.path.splitext(os.path.basename(args.input))[0]
        inferencer.process_audio_file(args.input, os.path.join(args.output, f"{name}_refined.mid"))
    elif os.path.isdir(args.input):
        exts = (".wav", ".mp3", ".flac", ".m4a")
        audio_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(args.input)
            for f in files
            if f.lower().endswith(exts)
        ]

        for audio_file in tqdm(audio_files, desc="Processing files"):
            try:
                name = os.path.splitext(os.path.basename(audio_file))[0]
                inferencer.process_audio_file(audio_file, os.path.join(args.output, f"{name}_refined.mid"))
            except Exception as exc:
                print(f"❌ Error processing {audio_file}: {exc}")
    else:
        raise ValueError(f"Invalid input path: {args.input}")


if __name__ == "__main__":
    main()
