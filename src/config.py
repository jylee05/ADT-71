# src/config.py
import os

class Config:
    # 1. 경로 설정 (Docker 환경)
    DATA_ROOT = "/public/e-gmd-v1.0.0"
    MERT_PATH = "m-a-p/MERT-v1-330M"
    CACHE_DIR = "./pretrained_models"
    
    # 2. 오디오 파라미터
    AUDIO_SR = 44100
    MERT_SR = 24000
    N_FFT = 2048
    HOP_LENGTH = 441
    N_MELS = 128
    FPS = AUDIO_SR // HOP_LENGTH  # = 100
    SEGMENT_SEC = 5.0
    
    # MERT parameters
    MERT_HOP = 320
    
    # 3. 모델 아키텍처
    DRUM_CHANNELS = 7
    FEATURE_DIM = 2
    HIDDEN_DIM = 512
    N_LAYERS = 6
    COND_LAYERS = 2
    N_HEADS = 8
    MERT_DIM = 1024
    MERT_LAYER_IDX = 10
    
    # 4. Dropout
    DROP_MERT_PROB = 0.15
    DROP_SPEC_PROB = 0.30
    DROP_PARTIAL_PROB = 0.5
    
    # 5. 학습 파라미터 (RTX 2080 1개 최적화)
    # MERT-330M: ~1.3GB (frozen)
    # 모델 + 그래디언트 + 옵티마이저: ~3-4GB
    # 남은 메모리로 배치 처리: 배치 10-12 권장
    BATCH_SIZE = 42
    GRAD_ACCUM_STEPS = 2  # 실제 배치는 12*7=84
    
    # Learning rate schedule
    LR_PEAK = 1e-4
    LR_MIN = 1e-6
    EPOCHS = 200
    WARMUP_EPOCHS = 5
    
    # Gradient clipping
    GRAD_CLIP_NORM = 1.0
    
    NUM_WORKERS = 16  # Docker 환경에 맞게 조정
    DEVICE = "cuda"  # GPU 1번만 사용
    
    # Flow matching loss
    C_MAX = 1.0
    C_MIN = 1e-4

    # -----------------------------
    # Imbalance / curriculum knobs
    # -----------------------------
    # WeightedRandomSampler multipliers per drum channel (KD, SN, HH, TT, CY, RD, BE).
    # (Used by EGMDTrainDataset._compute_sample_weights)
    OVERSAMPLE_KICK  = 1.0
    OVERSAMPLE_SNARE = 1.0
    OVERSAMPLE_HH    = 1.0
    OVERSAMPLE_TOMS  = 1.4
    OVERSAMPLE_CRASH = 1.6
    OVERSAMPLE_RIDE  = 1.4
    OVERSAMPLE_BELL  = 1.8

    # Count-strength factors per drum channel for log-count based scaling.
    # Larger values increase the impact of hit count for that channel.
    COUNT_STRENGTH_KICK  = 0.00
    COUNT_STRENGTH_SNARE = 0.00
    COUNT_STRENGTH_HH    = 0.00
    COUNT_STRENGTH_TOMS  = 0.20
    COUNT_STRENGTH_CRASH = 0.25
    COUNT_STRENGTH_RIDE  = 0.20
    COUNT_STRENGTH_BELL  = 0.30

    # Velocity-loss curriculum (applied inside AnnealedPseudoHuberLoss)
    # - start weaker so the model first learns "what" and "when"
    # - then ramp velocity importance to full strength.
    VEL_LAMBDA_START = 1.0
    VEL_LAMBDA_END = 1.0
