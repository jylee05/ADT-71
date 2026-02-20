#!/usr/bin/env python3
"""
N2N-Flow2 Inference Script (Refactored)
Matches evaluate.py logic: Stitching grids first, then peak picking.
"""

import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pretty_midi
from scipy.signal import find_peaks
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss

# Representative MIDI Notes for 7 drum classes (matching utils.py)
DRUM_MAPPING = [36, 38, 42, 47, 49, 51, 56]

class ADTInference:
    def __init__(self, args):
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        
        print(f"🎯 N2N-Flow2 Inference (Stitch-First Mode)")
        print(f"   Device: {self.device}")
        print(f"   Sampling Steps: {args.steps}")
        print(f"   Threshold: {args.threshold}")
        
        # Load model
        self.model = FlowMatchingTransformer(self.config).to(self.device)
        self.loss_fn = AnnealedPseudoHuberLoss(self.model, self.config).to(self.device)
        self.load_checkpoint(args.ckpt_path)
        
        self.model.eval()
        self.init_feature_extractors()

    def load_checkpoint(self, path):
        print(f"📂 Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict
            
        self.model.load_state_dict(new_state_dict)
        epoch = checkpoint.get('epoch', 'Unknown')
        print(f"   Loaded from epoch: {epoch}")

    def init_feature_extractors(self):
        self.mel_transform = MelSpectrogram(
            sample_rate=self.config.AUDIO_SR,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            normalized=True
        ).to(self.device)
        
        self.db_transform = AmplitudeToDB().to(self.device)
        self.resampler_to_44k = {}
        self.resampler_to_24k = {}

    def get_features(self, waveform_segment, sr):
        # 1. Mel-Spectrogram
        if sr != self.config.AUDIO_SR:
            if sr not in self.resampler_to_44k:
                self.resampler_to_44k[sr] = torchaudio.transforms.Resample(sr, self.config.AUDIO_SR).to(self.device)
            waveform_mel = self.resampler_to_44k[sr](waveform_segment).to(self.device)
        else:
            waveform_mel = waveform_segment.to(self.device)

        melspec = self.mel_transform(waveform_mel)
        melspec = self.db_transform(melspec)
        melspec = melspec.transpose(1, 2)

        # 2. MERT Waveform
        target_mert_sr = self.config.MERT_SR
        if sr != target_mert_sr:
            if sr not in self.resampler_to_24k:
                self.resampler_to_24k[sr] = torchaudio.transforms.Resample(sr, target_mert_sr).to(self.device)
            waveform_mert = self.resampler_to_24k[sr](waveform_segment.to(self.device))
        else:
            waveform_mert = waveform_segment.to(self.device)

        waveform_mert = waveform_mert.squeeze(0).unsqueeze(0)
        return waveform_mert, melspec

    def predict_segment(self, waveform_mert, spec):
        with torch.no_grad():
            predictions = self.loss_fn.sample(
                waveform_mert, spec, 
                steps=self.args.steps
            )
            return predictions[0].cpu().numpy()

    # [추가] evaluate.py의 핵심 로직인 Stitching 함수 구현
    def stitch_predictions(self, segment_results, total_duration):
        """Stitch overlapping segment predictions into a single full-length grid"""
        # 전체 길이에 맞는 빈 Grid 생성
        grid_length = int(total_duration * self.config.FPS) + 1
        drum_channels = self.config.DRUM_CHANNELS
        
        # (Frame, Channel * 2) 크기의 배열: 앞쪽은 Onset, 뒤쪽은 Velocity
        stitched_pred = np.full((grid_length, drum_channels * 2), -1.0, dtype=np.float32)
        overlap_count = np.zeros(grid_length, dtype=int)
        
        # 시간순 정렬
        segment_results.sort(key=lambda x: x['start_time'])
        
        for segment in segment_results:
            start_frame = int(segment['start_time'] * self.config.FPS)
            pred_grid = segment['pred_grid']
            
            # 현재 Grid가 전체 길이를 넘지 않도록 조정
            segment_length = min(pred_grid.shape[0], grid_length - start_frame)
            
            if segment_length > 0:
                for frame_idx in range(segment_length):
                    global_frame = start_frame + frame_idx
                    
                    if overlap_count[global_frame] == 0:
                        # 해당 프레임에 첫 데이터가 들어오면 그대로 대입
                        stitched_pred[global_frame] = pred_grid[frame_idx]
                    else:
                        # 이미 데이터가 있는 경우(Overlap 구간), 평균 계산 (Running Average)
                        alpha = 1.0 / (overlap_count[global_frame] + 1)
                        stitched_pred[global_frame] = (1 - alpha) * stitched_pred[global_frame] + alpha * pred_grid[frame_idx]
                    
                    overlap_count[global_frame] += 1
                    
        return stitched_pred

    def postprocess_predictions(self, predictions, onset_threshold=0.5):
        """
        [변경] 전체 곡 길이의 Grid를 받아서 한 번에 처리합니다.
        Evaluate.py의 calculate_file_metrics 로직과 유사하게 작동합니다.
        """
        # seq_len = predictions.shape[0]  <-- 사용되지 않음
        drum_channels = self.config.DRUM_CHANNELS
        
        # FIX: interleaved [on,vel,on,vel,...] -> (T, D, 2)
        pred_view = predictions.reshape(predictions.shape[0], drum_channels, 2)
        onset_pred = pred_view[:, :, 0]
        velocity_pred = pred_view[:, :, 1]

        # Velocity de-normalization: [-1, 1] -> [1, 127]
        velocity_norm = np.clip(((velocity_pred + 1) / 2) * 127, 1, 127)

        
        drum_events = []
        
        for drum_idx in range(drum_channels):
            # score(-1~1) 기준 threshold로 peak picking
            peaks, _ = find_peaks(
                onset_pred[:, drum_idx],
                height=onset_threshold,
                distance=int(0.05 * self.config.FPS)
            )
            
            for peak in peaks:
                time = peak / self.config.FPS  # 전체 곡 기준의 절대 시간
                velocity = int(velocity_norm[peak, drum_idx])
                midi_note = DRUM_MAPPING[drum_idx]
                onset_score = float(onset_pred[peak, drum_idx])
                
                drum_events.append({
                    'time': time,
                    'drum': drum_idx,
                    'midi_note': midi_note,
                    'velocity': velocity,
                    'onset_score': onset_score
                })
        return sorted(drum_events, key=lambda x: x['time'])

    def create_midi_file(self, drum_events, output_path, total_duration):
        pm = pretty_midi.PrettyMIDI()
        drum_program = pretty_midi.instrument_name_to_program('Synth Drum')
        drums = pretty_midi.Instrument(program=drum_program, is_drum=True)
        
        for event in drum_events:
            note = pretty_midi.Note(
                velocity=event['velocity'],
                pitch=event['midi_note'],
                start=event['time'],
                end=event['time'] + 0.1
            )
            drums.notes.append(note)
        
        pm.instruments.append(drums)
        pm.write(output_path)
        print(f"💾 MIDI saved: {output_path} ({len(drum_events)} notes)")

    def process_audio_file(self, input_path, output_path):
        print(f"🎵 Processing: {input_path}")
        
        waveform, sr = torchaudio.load(input_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        total_duration = waveform.shape[1] / sr
        segment_duration = self.config.SEGMENT_SEC
        segment_samples = int(segment_duration * sr)
        
        # [변경] Overlap 비율을 evaluate.py와 동일하게 0.5(50%)로 상향 조정
        # Stitching 효과를 극대화하기 위해 겹치는 구간을 늘림
        overlap_ratio = 0.5 
        hop_duration = segment_duration * (1 - overlap_ratio)
        
        num_segments = int(np.ceil((total_duration - segment_duration) / hop_duration)) + 1
        if total_duration <= segment_duration:
            num_segments = 1
        
        # [변경] 이벤트를 바로 추출하지 않고, Grid(예측 결과)를 모읍니다.
        segment_results = []
        
        for seg_idx in tqdm(range(num_segments), desc="Processing segments"):
            start_time = seg_idx * hop_duration
            start_sample = int(start_time * sr)
            end_sample = min(start_sample + segment_samples, waveform.shape[1])
            
            segment = waveform[:, start_sample:end_sample]
            
            if segment.shape[1] < segment_samples:
                pad_length = segment_samples - segment.shape[1]
                segment = F.pad(segment, (0, pad_length))
            
            waveform_mert, melspec = self.get_features(segment, sr)
            predictions = self.predict_segment(waveform_mert, melspec)
            
            # [변경] 예측된 Grid와 시작 시간을 저장
            segment_results.append({
                'start_time': start_time,
                'pred_grid': predictions
            })
            
        # [추가] 모든 세그먼트 처리가 끝난 후 Stitching 수행
        print("🔄 Stitching segments...")
        stitched_grid = self.stitch_predictions(segment_results, total_duration)
        
        # [추가] 합쳐진 전체 Grid에서 노트 추출 (Event Detection)
        print("🎹 extracting notes...")
        all_events = self.postprocess_predictions(stitched_grid, self.args.threshold)
        
        # MIDI 생성
        self.create_midi_file(all_events, output_path, total_duration)
        
        return all_events

def parse_args():
    parser = argparse.ArgumentParser(description='N2N-Flow2 Inference (Stitch-First)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for MIDI files')
    parser.add_argument('--steps', type=int, default=5, help='Number of sampling steps')
    parser.add_argument('--threshold', type=float, default=0.0, help='Onset score threshold (score in [-1, 1]; default 0.0)')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    inferencer = ADTInference(args)
    
    if os.path.isfile(args.input):
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(args.output, f"{input_name}_transcription.mid")
        inferencer.process_audio_file(args.input, output_path)
        
    elif os.path.isdir(args.input):
        import glob
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(args.input, f"**/*{ext}"), recursive=True))
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        for audio_file in tqdm(audio_files, desc="Processing files"):
            input_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(args.output, f"{input_name}_transcription.mid")
            try:
                inferencer.process_audio_file(audio_file, output_path)
            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
    else:
        print(f"❌ Invalid input path: {args.input}")

if __name__ == "__main__":
    main()
