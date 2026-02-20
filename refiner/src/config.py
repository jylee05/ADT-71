from dataclasses import dataclass


@dataclass
class RefinerTrainConfig:
    # data/window
    segment_sec: float = 20.0
    fps: int = 100

    # optimization
    # NOTE: batch_size and num_workers are inherited from src.config.Config at runtime.
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 1e-4

    # corruption for single-stage training
    p_extra_corrupt: float = 0.7
    p_add: float = 0.05
    p_delete: float = 0.08
    vel_noise_std: float = 0.10

    # loss weights
    w_edit: float = 1.0
    w_vel: float = 0.6
    w_identity: float = 0.2
    w_budget: float = 0.05

    # adaptive gate target (heuristic cleaness supervision)
    uncertainty_margin: float = 0.75

    # caching
    cache_dir: str = "./refiner/cache"

    # io
    save_dir: str = "./refiner/checkpoints"
