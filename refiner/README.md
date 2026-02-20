# Refiner (single-stage, FM frozen)

This folder contains a context-aware drum refiner designed from the discussion requirements:

- 20s window context.
- Input is frozen FM logits (typically sampled with `--fm_steps 3`).
- Focus on add/delete edits (no explicit shift head).
- Velocity is corrected with context-aware residual prediction.
- Adaptive clean/noisy gate is predicted by the model and can be overridden by user strength.
- Single-stage training: the refiner sees FM logits (optionally extra-corrupted) and GT targets together.
- One-file run path in `refiner/src/train.py` with integrated precompute/cache option.
- Multi-GPU for both FM precompute and refiner training (up to 4 GPUs by default).
- `batch_size` / `num_workers` follow existing FM repo config (`src.config.Config`).

## Run

```bash
python -m refiner.src.train --fm_ckpt /path/to/fm.pt --fm_steps 3 --precompute_cache --max_gpus 4
python -m refiner.src.eval --ckpt ./refiner/checkpoints/refiner_epoch_040.pt --cache_dir ./refiner/cache
```

## Inference from cached FM logits

```bash
python -m refiner.src.infer --refiner_ckpt ./refiner/checkpoints/refiner_epoch_040.pt --fm_logits_pt sample.pt --strength 1.0
```
