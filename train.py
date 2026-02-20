#!/usr/bin/env python3
"""
N2N-Flow2 Training Script
Enhanced version with improved data loading, cosine scheduling, and gradient clipping
"""

import os
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.config import Config
from src.dataset import EGMDTrainDataset, get_egmd_train_sampler
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.utils import seed_everything, get_cosine_schedule_with_warmup, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='N2N-Flow2 Training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--project_name', type=str, default='n2n-flow2', help='Wandb project name')
    parser.add_argument('--oversample', action='store_true', help='Use weighted oversampling for rare-hit files')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    
    # Setup
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.project_name,
            config=vars(config),
            name=f"n2n-flow2-{config.EPOCHS}epochs"
        )
    
    print(f"🚀 Starting N2N-Flow2 Training")
    print(f"   Device: {device}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Gradient Accumulation: {config.GRAD_ACCUM_STEPS}")
    print(f"   Learning Rate: {config.LR_PEAK} -> {config.LR_MIN}")
    print(f"   Gradient Clipping: {config.GRAD_CLIP_NORM}")
    
    # Dataset and DataLoader
    print("📁 Loading training dataset...")
    train_dataset = EGMDTrainDataset(compute_sample_weights=args.oversample)

    # Choose sampler by CLI option
    sampler = get_egmd_train_sampler(train_dataset, oversample=args.oversample)
    sampler_mode = "weighted oversampling (with replacement)" if args.oversample else "random sampling (without replacement)"
    print(f"   Sampler: {sampler_mode}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    # Model
    print("🏗️ Building model...")
    model = FlowMatchingTransformer(config).to(device)
    loss_fn = AnnealedPseudoHuberLoss(model, config).to(device)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"   Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        loss_fn.model = model
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR_PEAK,
        weight_decay=1e-4,
        betas=(0.9, 0.95)
    )
    
    # Calculate total steps (considering gradient accumulation) - Fix: use ceil to prevent scheduler bounce
    total_steps = math.ceil(len(train_loader) / config.GRAD_ACCUM_STEPS) * config.EPOCHS
    warmup_steps = int(config.WARMUP_EPOCHS * math.ceil(len(train_loader) / config.GRAD_ACCUM_STEPS))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=config.LR_MIN / config.LR_PEAK  # Ratio to base lr
    )
    
    # Resume training if checkpoint provided
    start_epoch = 0
    if args.resume:
        print(f"📂 Resuming from {args.resume}")
        saved_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch = saved_epoch  # saved_epoch is now 1-indexed, so start from it
    


    print(f"   Starting from epoch: {start_epoch + 1}")
    
    # Training loop
    print("🎯 Starting training...")
    model.train()
    
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        
        for batch_idx, (audio_mert, spec, target_grid) in enumerate(progress_bar):
            # Move to device
            audio_mert = audio_mert.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            target_grid = target_grid.to(device, non_blocking=True)
            
            # Initialize variables for progress bar (prevent UnboundLocalError)
            grad_norm = 0.0
            current_lr = optimizer.param_groups[0]['lr']
            
            # NaN/Inf check on inputs
            if torch.isnan(audio_mert).any() or torch.isinf(audio_mert).any():
                print(f"[WARNING] NaN/Inf detected in audio_mert at epoch {epoch+1}, batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            if torch.isnan(spec).any() or torch.isinf(spec).any():
                print(f"[WARNING] NaN/Inf detected in spec at epoch {epoch+1}, batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue
                
            if torch.isnan(target_grid).any() or torch.isinf(target_grid).any():
                print(f"[WARNING] NaN/Inf detected in target_grid at epoch {epoch+1}, batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Calculate training progress for annealing - Fix: match optimizer steps unit
            current_accum_step = (epoch * len(train_loader) + batch_idx) / config.GRAD_ACCUM_STEPS
            progress = current_accum_step / total_steps
            
            # Forward pass
            try:
                loss = loss_fn(audio_mert, spec, target_grid, progress)
                loss = loss.mean()  # Average over batch
                
                # NaN/Inf check on loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] NaN/Inf loss detected at epoch {epoch+1}, batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
            except Exception as e:
                print(f"[ERROR] Forward pass failed at epoch {epoch+1}, batch {batch_idx}: {e}")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Scale loss by accumulation steps
            loss = loss / config.GRAD_ACCUM_STEPS
            
            # Backward pass
            try:
                loss.backward()
            except Exception as e:
                print(f"[ERROR] Backward pass failed at epoch {epoch+1}, batch {batch_idx}: {e}")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Gradient accumulation
            if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0:
                # Check gradients for NaN/Inf
                if isinstance(model, nn.DataParallel):
                    params = model.module.parameters()
                else:
                    params = model.parameters()
                
                grad_norm = 0.0
                has_nan_grad = False
                for param in params:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                if has_nan_grad:
                    print(f"[WARNING] NaN/Inf gradients detected at epoch {epoch+1}, batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Gradient clipping
                if isinstance(model, nn.DataParallel):
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), config.GRAD_CLIP_NORM)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            epoch_loss += loss.item() * config.GRAD_ACCUM_STEPS
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item() * config.GRAD_ACCUM_STEPS:.4f}',
                'lr': f'{current_lr:.2e}',
                'progress': f'{progress:.2%}',
                'grad_norm': f'{grad_norm:.3f}'
            })
        
        # Process remainder gradients at epoch end
        if (len(train_loader) % config.GRAD_ACCUM_STEPS) != 0:
            # Check gradients for NaN/Inf
            if isinstance(model, nn.DataParallel):
                params = model.module.parameters()
            else:
                params = model.parameters()
            
            has_nan_grad = False
            for param in params:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if not has_nan_grad:
                # Gradient clipping
                if isinstance(model, nn.DataParallel):
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), config.GRAD_CLIP_NORM)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                
                optimizer.step()
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
            if args.wandb:
                wandb.log({
                    'train_loss': loss.item() * config.GRAD_ACCUM_STEPS,
                    'learning_rate': current_lr,
                    'training_progress': progress,
                    'gradient_norm': grad_norm,
                    'epoch': epoch + 1
                })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_epoch_loss, checkpoint_path)
        
        # Log epoch summary to wandb
        if args.wandb:
            wandb.log({
                'epoch_loss': avg_epoch_loss,
                'epoch': epoch + 1
            })
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, f"final_checkpoint_epoch_{config.EPOCHS}.pt")
    save_checkpoint(model, optimizer, scheduler, config.EPOCHS, avg_epoch_loss, final_checkpoint_path)
    
    print("🎉 Training completed!")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()