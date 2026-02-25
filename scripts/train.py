#!/usr/bin/env python3
"""
Training script for discrete diffusion language model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --resume checkpoints/checkpoint_step_005000.pt
"""

import argparse
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusionllm.model import ModelConfig, DiscreteDiffusionTransformer
from diffusionllm.diffusion import forward_diffusion, compute_loss, get_noise_schedule
from diffusionllm.utils import (
    get_lr_schedule,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    AverageMeter,
    TrainingLogger,
)


@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Diffusion process
    T: int = 1000
    
    # Model architecture
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    
    # Training hyperparameters
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    seq_len: int = 128
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Optimization
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Logging & checkpointing
    log_every: int = 100
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Special tokens
    mask_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2
    
    # Data
    data_path: str = "data/train.txt"
    num_workers: int = 4
    
    # Mixed precision
    use_amp: bool = True
    
    # Optional: Use wandb/tensorboard
    use_wandb: bool = False
    use_tensorboard: bool = False


class SimpleTextDataset:
    """
    Simple text dataset for training.
    
    Replace this with your actual dataset implementation.
    """
    
    def __init__(self, data_path: str, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> torch.Tensor:
        """Load and tokenize text data."""
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                lines = f.readlines()
            
            # Simple tokenization (replace with proper tokenizer)
            all_tokens = []
            for line in lines:
                tokens = [min(ord(c) % (self.vocab_size - 3) + 3, self.vocab_size - 1) 
                         for c in line.strip()]
                all_tokens.extend(tokens)
                all_tokens.append(self.eos_token_id if hasattr(self, 'eos_token_id') else 2)
            
            # Create fixed-length sequences
            data = []
            for i in range(0, len(all_tokens) - self.seq_len, self.seq_len):
                data.append(all_tokens[i:i + self.seq_len])
            
            return torch.tensor(data, dtype=torch.long)
        else:
            print(f"Data file not found: {data_path}. Using synthetic data.")
            return torch.randint(3, self.vocab_size, (10000, self.seq_len))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def train_step(
    model: torch.nn.Module,
    x0: torch.Tensor,
    train_config: TrainConfig,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    alpha: Optional[torch.Tensor] = None,
) -> tuple:
    """Single training step."""
    model.train()
    batch_size = x0.shape[0]
    
    # Sample random timestep
    t = torch.randint(0, train_config.T, (batch_size,), device=device)
    
    # Forward diffusion
    xt, mask = forward_diffusion(
        x0, t, train_config.T, 
        model_config.mask_token_id, 
        model_config.pad_token_id,
        alpha=alpha,
    )
    
    # Forward pass with mixed precision
    optimizer.zero_grad()
    
    with autocast('cuda', enabled=train_config.use_amp):
        logits = model(xt, t)
        loss = compute_loss(
            logits, x0, mask, t, train_config.T,
            model_config.pad_token_id,
            alpha=alpha,
        )
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), train_config.grad_clip
    )
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # Compute perplexity
    with torch.no_grad():
        with autocast('cuda', enabled=train_config.use_amp):
            if mask.sum() > 0:
                logits_masked = logits[mask]
                x0_masked = x0[mask]
                ce_loss = torch.nn.functional.cross_entropy(logits_masked, x0_masked)
                perplexity = torch.exp(ce_loss)
            else:
                perplexity = torch.tensor(float('inf'), device=device)
    
    return loss, perplexity, grad_norm


def main(args):
    """Main training function."""
    # Setup configuration
    train_config = TrainConfig()
    model_config = ModelConfig(
        vocab_size=train_config.vocab_size,
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        dropout=train_config.dropout,
        max_seq_len=train_config.max_seq_len,
        mask_token_id=train_config.mask_token_id,
        pad_token_id=train_config.pad_token_id,
        eos_token_id=train_config.eos_token_id,
    )
    
    # Override from config file if provided
    if args.config:
        try:
            from omegaconf import OmegaConf
            conf = OmegaConf.load(args.config)
            for key, value in asdict(conf).items():
                if hasattr(train_config, key):
                    setattr(train_config, key, value)
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        except ImportError:
            print("omegaconf not installed, ignoring config file")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize model
    model = DiscreteDiffusionTransformer(model_config).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda', enabled=train_config.use_amp)
    
    # Get noise schedule
    alpha = get_noise_schedule(train_config.T)
    
    # Load checkpoint if resuming
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step, model, optimizer, scaler = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        start_step += 1
        print(f"Resuming from step {start_step}")
    
    # Setup data
    dataset = SimpleTextDataset(
        data_path=train_config.data_path,
        seq_len=train_config.seq_len,
        vocab_size=train_config.vocab_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )
    
    # Setup logging
    logger = TrainingLogger(
        log_dir=train_config.log_dir,
        use_wandb=train_config.use_wandb,
        use_tensorboard=train_config.use_tensorboard,
    )
    
    # Training loop
    step = start_step
    loss_meter = AverageMeter("loss")
    ppl_meter = AverageMeter("ppl")
    
    print(f"Starting training from step {step} to {train_config.max_steps}")
    
    while step < train_config.max_steps:
        for batch in dataloader:
            if step >= train_config.max_steps:
                break
            
            x0 = batch.to(device)
            
            # Training step
            loss, perplexity, grad_norm = train_step(
                model, x0, train_config, model_config,
                optimizer, scaler, device, alpha
            )
            
            # Update meters
            loss_meter.update(loss.item())
            if torch.isfinite(perplexity):
                ppl_meter.update(perplexity.item())
            
            # Update learning rate
            lr_mult = get_lr_schedule(step, train_config.warmup_steps, train_config.max_steps)
            current_lr = train_config.lr * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Logging
            if step % train_config.log_every == 0:
                metrics = {
                    'train/loss': loss_meter.avg,
                    'train/perplexity': ppl_meter.avg,
                    'train/grad_norm': grad_norm,
                    'train/lr': current_lr,
                }
                logger.log(metrics, step=step)
                
                # Reset meters
                loss_meter.reset()
                ppl_meter.reset()
            
            # Checkpointing
            if step % train_config.save_every == 0:
                checkpoint_path = os.path.join(
                    train_config.checkpoint_dir, 
                    f"checkpoint_step_{step:06d}.pt"
                )
                save_checkpoint(
                    step, model, optimizer, scaler,
                    asdict(train_config), checkpoint_path
                )
            
            step += 1
    
    # Save final checkpoint
    final_path = os.path.join(train_config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(
        step - 1, model, optimizer, scaler,
        asdict(train_config), final_path
    )
    
    logger.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Discrete Diffusion Language Model")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to training data"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory to save checkpoints"
    )
    
    args = parser.parse_args()
    
    # Override config from command line
    if args.data_path:
        TrainConfig.data_path = args.data_path
    if args.checkpoint_dir:
        TrainConfig.checkpoint_dir = args.checkpoint_dir
    
    main(args)
