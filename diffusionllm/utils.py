"""
Utility functions for discrete diffusion language model.

Includes checkpointing, learning rate scheduling, and other helpers.
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler


def get_lr_schedule(step: int, warmup_steps: int, max_steps: int) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
    
    Returns:
        Learning rate multiplier (multiply by base lr)
    """
    if step < warmup_steps:
        # Linear warmup
        return step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: dict,
    path: str,
    save_optimizer: bool = True,
):
    """
    Save training checkpoint.
    
    Args:
        step: Current training step
        model: Model to save
        optimizer: Optimizer to save
        scaler: AMP GradScaler to save
        config: Configuration dict
        path: Path to save checkpoint
        save_optimizer: Whether to save optimizer state
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'config': config,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    
    if save_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    device: Optional[torch.device] = None,
    load_optimizer: bool = True,
) -> Tuple[int, nn.Module, Optional[torch.optim.Optimizer], Optional[GradScaler]]:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scaler: GradScaler to load state into (optional)
        device: Device to load tensors to
        load_optimizer: Whether to load optimizer state
    
    Returns:
        step: Training step checkpoint was saved at
        model: Model with loaded weights
        optimizer: Optimizer with loaded state (if provided)
        scaler: Scaler with loaded state (if provided)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if load_optimizer and scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Restore RNG state for reproducibility
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    
    return checkpoint['step'], model, optimizer, scaler


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class TrainingLogger:
    """Logger for training metrics with optional wandb/tensorboard support."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
    ):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.step = 0
        
        # Initialize tensorboard if requested
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Initialize wandb if requested
        if use_wandb:
            import wandb
            wandb.init(project="diffusion-llm")
            self.wandb = wandb
        else:
            self.wandb = None
    
    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to all enabled backends."""
        if step is not None:
            self.step = step
        
        # Console logging
        log_str = f"Step {self.step:6d} | "
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_str += f"{name}: {value:.4f} | "
        print(log_str)
        
        # Tensorboard logging
        if self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(name, value, self.step)
        
        # WandB logging
        if self.wandb is not None:
            self.wandb.log(metrics, step=self.step)
    
    def close(self):
        """Close loggers."""
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()


def setup_distributed_training():
    """
    Setup for distributed training with DDP.
    
    Returns:
        rank: Process rank
        world_size: Total number of processes
        device: Device for this process
    """
    if not torch.distributed.is_available():
        return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    return rank, world_size, device


def is_master_process(rank: int = 0) -> bool:
    """Check if current process is the master process."""
    return rank == 0
