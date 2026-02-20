# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from transformers import Wav2Vec2Model

# -------------------------------------------------------------------
# Positional Embedding (Time Step용) - Diffusion/Flow Matching의 t 임베딩
# -------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # 극단값 방지
        x_clamped = torch.clamp(x, min=-10.0, max=10.0)
        
        emb = x_clamped[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# -------------------------------------------------------------------
# Sequence Positional Embedding (시퀀스 위치 정보용)
# -------------------------------------------------------------------
class SequencePositionalEncoding(nn.Module):
    """Transformer 시퀀스용 Dynamic Sinusoidal Positional Encoding"""
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # 기본 길이만 미리 계산 (메모리 절약)
        self._create_pe(max_len)
    
    def _create_pe(self, length, device=None):
        """동적으로 positional encoding 생성"""
        pe = torch.zeros(length, self.d_model, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, length, d_model)

        # register once; later dynamic expansions should update the existing buffer
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (Batch, Seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # 현재 테이블보다 긴 시퀀스가 들어오면 동적으로 확장
        if seq_len > self.pe.size(1):
            print(f"[INFO] Expanding positional encoding from {self.pe.size(1)} to {seq_len}")
            self._create_pe(seq_len * 2, device=x.device)  # 여유있게 2배로 확장
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# -------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation) Layer
# -------------------------------------------------------------------
class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, condition):
        """
        x: (Batch, Seq, Dim)
        condition: (Batch, Cond_Dim)
        """
        # Scale과 Shift 계수 예측
        params = self.proj(condition)
        scale, shift = params.chunk(2, dim=-1)
        
        # Scale 값을 제한하여 gradient explosion 방지
        scale = torch.tanh(scale) * 0.5
        
        # 차원 맞추기 (Broadcasting)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        # Affine 변환
        return x * (1 + scale) + shift

# -------------------------------------------------------------------
# FiLM 기반 Transformer Decoder Layer
# -------------------------------------------------------------------
class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        # Attention Layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Normalization Layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # FiLM Layers
        self.film1 = FiLMLayer(d_model, d_model)
        self.film2 = FiLMLayer(d_model, d_model)
        self.film3 = FiLMLayer(d_model, d_model)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, tgt, memory, cond_emb):
        """
        tgt: Decoder 입력 (Noisy Grid)
        memory: Cross Attention용 Context (Spec + MERT)
        cond_emb: FiLM용 Global Condition (Time + Audio Summary)
        """
        # 1. Self Attention Block
        tgt2 = self.norm1(tgt)
        tgt2 = self.film1(tgt2, cond_emb)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2)
        tgt = tgt + self.dropout(tgt2)
        
        # 2. Cross Attention Block
        tgt2 = self.norm2(tgt)
        tgt2 = self.film2(tgt2, cond_emb)
        tgt2, _ = self.cross_attn(tgt2, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        
        # 3. Feed Forward Block
        tgt2 = self.norm3(tgt)
        tgt2 = self.film3(tgt2, cond_emb)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

# -------------------------------------------------------------------
# 메인 모델: Flow Matching Transformer
# -------------------------------------------------------------------
class FlowMatchingTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.DRUM_CHANNELS * config.FEATURE_DIM
        
        # 1. Input Projection
        self.proj_in = nn.Linear(self.input_dim, config.HIDDEN_DIM)
        
        # 2. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.HIDDEN_DIM),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        # 3. Sequence Positional Encodings (5-second segments with sliding window)
        self.pos_enc_main = SequencePositionalEncoding(config.HIDDEN_DIM, max_len=2000)  # ~20 seconds at 100 FPS
        self.pos_enc_mert = SequencePositionalEncoding(config.HIDDEN_DIM, max_len=2000)
        self.pos_enc_spec = SequencePositionalEncoding(config.HIDDEN_DIM, max_len=2000)
        
        # 4. Condition Encoders
        # 4-1. MERT Path
        self.mert_proj = nn.Linear(config.MERT_DIM, config.HIDDEN_DIM)
        self.mert_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=4, 
                dim_feedforward=config.HIDDEN_DIM*2,
                batch_first=True, 
                norm_first=True
            ),
            num_layers=config.COND_LAYERS
        )

        # 4-2. Spectrogram Path
        self.spec_proj = nn.Linear(config.N_MELS, config.HIDDEN_DIM)
        self.spec_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=4, 
                dim_feedforward=config.HIDDEN_DIM*2,
                batch_first=True, 
                norm_first=True
            ),
            num_layers=config.COND_LAYERS
        )
        
        # 5. Learned Null Embeddings
        self.null_mert_emb = nn.Parameter(torch.randn(1, 1, config.MERT_DIM) * 0.02)
        self.null_spec_emb = nn.Parameter(torch.randn(1, 1, config.N_MELS) * 0.02)
        
        # 6. Main Decoder Layers
        self.layers = nn.ModuleList([
            FiLMTransformerDecoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=config.N_HEADS, 
                dim_feedforward=config.HIDDEN_DIM * 4
            )
            for _ in range(config.N_LAYERS)
        ])
        
        # 7. Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, self.input_dim)
        
        # 8. Weight Initialization (MERT 로드 전에 수행해 사전학습 가중치 보존)
        self._init_weights()

        # 9. Pretrained MERT Model Load
        print(f"Loading MERT from {config.MERT_PATH}...")
        self.mert = Wav2Vec2Model.from_pretrained(config.MERT_PATH, cache_dir=config.CACHE_DIR)
        self.mert.eval()
        for p in self.mert.parameters():
            p.requires_grad = False
    
    def train(self, mode: bool = True):
        """Keep pretrained MERT in eval mode even during outer model training."""
        super().train(mode)
        self.mert.eval()
        return self

    def _init_weights(self):
        """Xavier/Kaiming 초기화로 gradient explosion 방지"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Null embeddings는 더 작게 초기화
        nn.init.normal_(self.null_mert_emb, 0, 0.01)
        nn.init.normal_(self.null_spec_emb, 0, 0.01)
    
    def extract_mert(self, audio):
        """MERT 모델에서 특정 레이어 특징 추출"""
        with torch.no_grad():
            # 입력 오디오 정규화 (MERT best practice)
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                print("[WARNING] NaN/Inf in audio input to MERT")
                audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Audio standardization for better MERT performance
            # Replace problematic L2 normalization with proper standardization
            audio_mean = audio.mean(dim=-1, keepdim=True)
            audio_std = audio.std(dim=-1, keepdim=True) + 1e-6
            audio_norm = (audio - audio_mean) / audio_std
            
            outputs = self.mert(audio_norm, output_hidden_states=True)
            features = outputs.hidden_states[self.config.MERT_LAYER_IDX]
            
            # MERT 출력 체크
            if torch.isnan(features).any():
                print("[WARNING] NaN in MERT features")
                features = torch.nan_to_num(features, nan=0.0)
                
            return features

    def apply_condition_dropout(self, feat, null_emb, drop_prob, partial_prob):
        """
        Dropouts:
        1. Complete Dropout: 배치 내 특정 샘플의 전체 컨디션을 날림
        2. Partial Dropout: 시간축의 일부 구간을 마스킹
        """
        if not self.training:
            return feat
        
        B, T, D = feat.shape
        device = feat.device
        out_feat = feat.clone()
        
        # 1. Complete Dropout
        drop_mask = torch.bernoulli(torch.full((B, 1, 1), drop_prob, device=device)).bool()
        out_feat = torch.where(drop_mask, null_emb, out_feat)
        
        # 2. Partial Dropout
        if random.random() < partial_prob:
            mask_ratio = random.uniform(0.1, 0.5)
            mask_len = int(T * mask_ratio)
            
            if mask_len > 0:
                start = random.randint(0, T - mask_len)
                out_feat[:, start:start+mask_len, :] = null_emb

        return out_feat

    def forward(self, x_t, t, audio_mert, spec_feats):
        """
        Args:
            x_t: Noisy drum grid (B, T, D)
            t: Time step (B,)
            audio_mert: Raw audio for MERT (B, samples)
            spec_feats: Mel-spectrogram features (B, T, N_MELS)
        """
        # 0. MERT Feature Extraction
        mert_feats = self.extract_mert(audio_mert)

        # 1. MERT를 Spec 길이에 맞게 Interpolate (dropout 전에 길이 정렬)
        target_len = spec_feats.shape[1]
        if mert_feats.shape[1] != target_len:
            mert_feats = mert_feats.permute(0, 2, 1)
            mert_feats = F.interpolate(mert_feats, size=target_len, mode='linear', align_corners=False)
            mert_feats = mert_feats.permute(0, 2, 1)

        # 2. Condition Dropout & Substitution
        mert_h = self.apply_condition_dropout(
            mert_feats, self.null_mert_emb,
            self.config.DROP_MERT_PROB, self.config.DROP_PARTIAL_PROB
        )
        spec_h = self.apply_condition_dropout(
            spec_feats, self.null_spec_emb,
            self.config.DROP_SPEC_PROB, self.config.DROP_PARTIAL_PROB
        )

        # 3. Project to hidden dim
        mert_emb = self.mert_proj(mert_h)
        spec_emb = self.spec_proj(spec_h)
        
        # 4. Positional Encoding
        mert_emb = self.pos_enc_mert(mert_emb)
        spec_emb = self.pos_enc_spec(spec_emb)
        
        # 5. Encode Separately
        mert_emb = self.mert_encoder(mert_emb)
        spec_emb = self.spec_encoder(spec_emb)
        
        # 6. Concatenate for Cross Attention
        memory = torch.cat([mert_emb, spec_emb], dim=1)
        
        # Global Condition for FiLM
        time_emb = self.time_mlp(t)
        
        # Time embedding NaN 체크
        if torch.isnan(time_emb).any():
            print("[WARNING] NaN in time_emb")
            time_emb = torch.nan_to_num(time_emb, nan=0.0)
        
        # 오디오 전체 맥락
        audio_ctx = memory.mean(dim=1)
        
        # Audio context NaN 체크  
        if torch.isnan(audio_ctx).any():
            print("[WARNING] NaN in audio_ctx")
            audio_ctx = torch.nan_to_num(audio_ctx, nan=0.0)
            
        cond_emb = time_emb + audio_ctx
        
        # 8. Main Network Flow
        h = self.proj_in(x_t)
        
        # proj_in 출력 NaN 체크
        if torch.isnan(h).any():
            print("[WARNING] NaN in proj_in output")
            h = torch.nan_to_num(h, nan=0.0)
        
        h = self.pos_enc_main(h)
        h = h + time_emb.unsqueeze(1)
        
        for layer in self.layers:
            h = layer(h, memory, cond_emb)
            # 각 layer 후 NaN 체크
            if torch.isnan(h).any():
                print("[WARNING] NaN in transformer layer output")
                h = torch.nan_to_num(h, nan=0.0)
                break
            
        return self.head(h)

# -------------------------------------------------------------------
# Loss Wrapper: Annealed Pseudo-Huber Loss
# -------------------------------------------------------------------
class AnnealedPseudoHuberLoss(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config


        # Per-class weights (Kick, Snare, HH, Toms, Crash, Ride, Bell)
        # NOTE: These are *starting* weights; you can tune later.
        self.class_weights = torch.tensor(
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32
        )

        # Velocity curriculum (applies only on hit frames)
        # Start with weaker velocity loss, gradually ramp up.
        self.vel_lambda_start = getattr(config, 'VEL_LAMBDA_START', 1.0)
        self.vel_lambda_end = getattr(config, 'VEL_LAMBDA_END', 1.0)

    def get_c(self, progress):
        """Annealing: 학습 초기에는 큰 c(MSE처럼), 후반에는 작은 c(L1처럼)"""
        alpha = progress
        return (1 - alpha) * self.config.C_MAX + alpha * self.config.C_MIN
    
    @staticmethod
    def _sample_base_noise(ref):
        """Flow matching base distribution: standard Gaussian N(0, 1)."""
        return torch.randn_like(ref)

    def sample_time(self, batch_size, device):
        """Flow Matching Time Sampling"""
        eps = 1e-4

        t = torch.rand(batch_size, device = device)

        t = t * (1 - 2 * eps) + eps
        return t

    def forward(self, audio_mert, spec, target_score, progress):
        """
        Args:
            audio_mert: Raw audio for MERT (B, samples)
            spec: Mel-spectrogram (B, T, N_MELS)
            target_score: Ground truth drum grid (B, T, D)
            progress: Training progress [0, 1]
        """
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        # Flow Matching Setup
        t = self.sample_time(batch_size, device)
        
        x_1 = target_score        
        x_0 = self._sample_base_noise(x_1)
        
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        t_view = t.view(batch_size, 1, 1)
        x_t = (1 - t_view) * x_0 + t_view * x_1
        
        # Velocity Prediction
        pred_v = self.model(x_t, t, audio_mert, spec)
        
        # Target velocity
        target_v = x_1 - x_0 
        
        # Annealed Pseudo-Huber Loss (element-wise)
        diff = pred_v - target_v
        c = self.get_c(progress)
        loss = torch.sqrt(diff.pow(2) + c**2) - c  # (B, T, D)

        # ---- weighting / masking ----
        # Device sync for weights
        if self.class_weights.device != loss.device:
            self.class_weights = self.class_weights.to(loss.device)

        B, T, D = target_score.shape
        C = self.config.DRUM_CHANNELS

        # (B, T, C, 2)
        loss_view = loss.view(B, T, C, 2)
        target_view = target_score.contiguous().view(B, T, C, 2)

        # Hit mask 기준: onset label > -0.5  (labels are -1 for silence, +1 for hit)
        onset_hit = (target_view[..., 0] > -0.5).float()  # (B, T, C)

        # Per-class weights broadcast: (1, 1, C)
        w = self.class_weights.view(1, 1, C)

        # Onset weight: only boost positives (hit frames)
        onset_weight = 1.0 + onset_hit * (w - 1.0)  # (B, T, C)

        # Velocity curriculum:
        # - non-hit frames also keep base weight 1.0 (onset loss와 동일한 기본 가중치)
        # - hit frames get additional class-aware weighting that ramps up over training

        vel_lambda = self.vel_lambda_start + float(progress) * (self.vel_lambda_end - self.vel_lambda_start)
        vel_weight = 1.0 + vel_lambda * onset_hit * (onset_weight - 1.0)  # (B, T, C)

        # Apply
        onset_loss = loss_view[..., 0] * onset_weight
        vel_loss = loss_view[..., 1] * vel_weight

        weighted_loss = onset_loss + vel_loss  # (B, T, C)
        return weighted_loss.mean(dim=[1, 2])

    @torch.no_grad()
    def sample(self, audio_mert, spec, steps=10, init_score=None, start_t=0.0):
        """Inference용 샘플링"""
        self.model.eval()
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        if isinstance(self.model, nn.DataParallel):
            input_dim = self.model.module.input_dim
        else:
            input_dim = self.model.input_dim

        seq_len = spec.size(1) 
        
        if init_score is not None and start_t > 0:
            noise = self._sample_base_noise(init_score)
            x_t = start_t * init_score + (1 - start_t) * noise
            t_current = start_t
        else:
            x_t = self._sample_base_noise(
                torch.empty(batch_size, seq_len, input_dim, device=device)
            )
            t_current = 0.0
            
        steps_to_run = int(steps * (1.0 - t_current))
        if steps_to_run < 1: 
            steps_to_run = 1
        
        dt = (1.0 - t_current) / steps_to_run
        
        for i in range(steps_to_run):
            t_val = t_current + i * dt
            t_tensor = torch.full((batch_size,), t_val, device=device)

            # k1
            v1 = self.model(x_t, t_tensor, audio_mert, spec)
            x_euler = x_t + v1 * dt

            # k2 at t+dt
            t_tensor_next = torch.full((batch_size,), t_val + dt, device=device)
            v2 = self.model(x_euler, t_tensor_next, audio_mert, spec)

            # Heun update
            x_t = x_t + 0.5 * (v1 + v2) * dt
            
        x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t
