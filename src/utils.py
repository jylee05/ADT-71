# src/utils.py
import torch
import numpy as np
import random
import os
import math

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset의 매핑을 역으로 이용하여 MIDI 생성 시 사용
REVERSE_DRUM_MAPPING = {
    0: 36,  # Kick
    1: 38,  # Snare
    2: 42,  # Hi-hat (Closed)
    3: 47,  # Tom (Mid-Tom)
    4: 49,  # Crash
    5: 51,  # Ride
    6: 56   # Bell (Cowbell)
}

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """Cosine annealing scheduler with warmup"""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase: linear increase
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr and 1.0 (which will be multiplied by base lr)
        return min_lr + cosine_decay * (1.0 - min_lr)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'train_loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, scheduler, filepath):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    # Handle DataParallel vs regular model
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {filepath} (epoch {epoch + 1})")
    return epoch, loss