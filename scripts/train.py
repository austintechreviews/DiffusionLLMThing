#!/usr/bin/env python3
"""
Production training script for discrete diffusion language model.

Features:
- Tokenizer integration
- Validation loop with perplexity tracking
- Early stopping
- Gradient accumulation
- WandB/TensorBoard logging
- Model presets

Usage:
    python scripts/train.py --data-dir data/processed --model-preset base
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusionllm.config import ModelConfig, MODEL_PRESETS, get_model_config, print_model_summary
from diffusionllm.model import DiscreteDiffusionTransformer
from diffusionllm.diffusion import forward_diffusion, compute_loss, get_noise_schedule
from diffusionllm.data import TokenizedDataset, create_dataloader, load_datasets
from diffusionllm.tokenizer import DiffusionTokenizer
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
    
    # Data
    data_dir: str = "data/processed"
    tokenizer_path: str = ""
    
    # Model
    model_preset: str = "base"
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    use_rotary_embeddings: bool = False
    
    # Diffusion
    T: int = 1000
    noise_schedule: str = "linear"
    
    # Training hyperparameters
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_seq_len: int = 512
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Optimization
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Validation
    val_every: int = 1000
    val_batches: int = 100
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Logging & checkpointing
    log_every: int = 100
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Mixed precision
    use_amp: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "diffusion-llm"
    use_tensorboard: bool = False
    
    # Reproducibility
    seed: int = 42
    
    # Special tokens (usually from tokenizer)
    mask_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2
    bos_token_id: int = 3


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    alpha: torch.Tensor,
    max_batches: int = 100,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    num_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if num_batches >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            batch_size = input_ids.shape[0]
            
            # Sample timesteps
            t = torch.randint(0, train_config.T, (batch_size,), device=device)
            
            # Forward diffusion
            xt, mask = forward_diffusion(
                input_ids, t, train_config.T,
                model_config.mask_token_id,
                model_config.pad_token_id,
                alpha=alpha,
            )
            
            # Forward pass
            logits = model(xt, t)
            
            # Compute loss
            loss = compute_loss(
                logits, input_ids, mask, t, train_config.T,
                model_config.pad_token_id,
                alpha=alpha,
            )
            
            # Compute raw cross-entropy (for perplexity)
            if mask.sum() > 0:
                ce_loss = torch.nn.functional.cross_entropy(
                    logits[mask], input_ids[mask]
                )
                total_ce_loss += ce_loss.item() * mask.sum().item()
                num_tokens += mask.sum().item()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    avg_ce_loss = total_ce_loss / max(1, num_tokens)
    perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()
    
    model.train()
    
    return {
        "val/loss": avg_loss,
        "val/cross_entropy": avg_ce_loss,
        "val/perplexity": perplexity,
    }


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    model_config: ModelConfig,
    train_config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    alpha: Optional[torch.Tensor],
    step: int,
) -> tuple:
    """Single training step with gradient accumulation."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    batch_size = input_ids.shape[0]
    use_amp = device.type == 'cuda' and train_config.use_amp
    
    # Sample random timestep
    t = torch.randint(0, train_config.T, (batch_size,), device=device)
    
    # Forward diffusion
    xt, mask = forward_diffusion(
        input_ids, t, train_config.T,
        model_config.mask_token_id,
        model_config.pad_token_id,
        alpha=alpha,
    )
    
    # Forward pass with mixed precision
    with autocast('cuda', enabled=use_amp):
        logits = model(xt, t)
        loss = compute_loss(
            logits, input_ids, mask, t, train_config.T,
            model_config.pad_token_id,
            alpha=alpha,
        )
    
    # Scale loss for gradient accumulation
    loss = loss / train_config.grad_accum_steps
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Optimizer step (only after grad_accum_steps)
    grad_norm = torch.tensor(0.0)
    if (step + 1) % train_config.grad_accum_steps == 0:
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        )
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Compute perplexity
    perplexity = torch.tensor(float('inf'), device=device)
    if mask.sum() > 0:
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                ce_loss = torch.nn.functional.cross_entropy(logits[mask], input_ids[mask])
                perplexity = torch.exp(ce_loss)
    
    return loss.item() * train_config.grad_accum_steps, perplexity, grad_norm


def main(args):
    """Main training function."""
    # Load configuration
    train_config = TrainConfig()

    # Apply preset
    if args.model_preset:
        model_config = get_model_config(
            args.model_preset,
            use_rotary_embeddings=args.use_rotary_embeddings,
        )
    else:
        model_config = ModelConfig(
            vocab_size=train_config.vocab_size,
            hidden_dim=train_config.hidden_dim,
            num_layers=train_config.num_layers,
            num_heads=train_config.num_heads,
            dropout=train_config.dropout,
            max_seq_len=train_config.max_seq_len,
            use_rotary_embeddings=train_config.use_rotary_embeddings,
        )

    # Override from config file
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

    # Override from command line
    if args.data_dir:
        train_config.data_dir = args.data_dir
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir
    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.lr:
        train_config.lr = args.lr

    # Set seed
    set_seed(train_config.seed)

    # Setup device
    device = torch.device('cpu')
    use_cuda = False
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1).cuda()
            y = (x + x).item()
            device = torch.device('cuda')
            use_cuda = True
            print("Training on device: cuda")
        except Exception:
            print("CUDA not usable, falling back to CPU")
    
    if not use_cuda:
        device = torch.device('cpu')
        train_config.use_amp = False  # Disable AMP on CPU
    
    # Adjust model max_seq_len based on data BEFORE creating model
    if train_config.data_dir:
        metadata_path = os.path.join(train_config.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            data_max_len = metadata.get('max_seq_len', 0)
            if data_max_len > 0 and data_max_len < model_config.max_seq_len:
                # Add small buffer for BOS/EOS
                new_max_len = min(data_max_len + 4, model_config.max_seq_len)
                print(f"Adjusting model max_seq_len from {model_config.max_seq_len} to {new_max_len} (data max: {data_max_len})")
                model_config.max_seq_len = new_max_len
    
    # Print model summary
    print_model_summary(model_config)
    
    # Load tokenizer
    tokenizer_path = train_config.tokenizer_path or os.path.join(train_config.data_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = DiffusionTokenizer.load(tokenizer_path)
        model_config.vocab_size = tokenizer.actual_vocab_size
        train_config.mask_token_id = tokenizer.mask_token_id
        train_config.pad_token_id = tokenizer.pad_token_id
        train_config.eos_token_id = tokenizer.eos_token_id
        train_config.bos_token_id = tokenizer.bos_token_id
        print(f"  Vocabulary size: {tokenizer.actual_vocab_size}")
    else:
        print(f"Tokenizer not found at {tokenizer_path}, using default special tokens")
        tokenizer = None

    # Initialize model
    model = DiscreteDiffusionTransformer(model_config).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    
    # Gradient scaler
    scaler = GradScaler('cuda', enabled=use_cuda and train_config.use_amp)
    
    # Get noise schedule
    alpha = get_noise_schedule(train_config.T, schedule=train_config.noise_schedule)
    
    # Load datasets
    print(f"\nLoading data from {train_config.data_dir}")
    dataloaders = load_datasets(
        train_config.data_dir,
        max_length=train_config.max_seq_len,
        batch_size=train_config.batch_size,
        num_workers=0,  # Set >0 for faster loading
        cache_in_memory=True,
    )
    
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('val')
    
    if train_loader is None:
        print("Error: No training data found")
        sys.exit(1)
    
    # Setup logging
    logger = TrainingLogger(
        log_dir=train_config.log_dir,
        use_wandb=train_config.use_wandb,
        use_tensorboard=train_config.use_tensorboard,
    )
    
    if train_config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_config.wandb_project,
                config=asdict(train_config),
            )
        except ImportError:
            print("wandb not installed, disabling")
            train_config.use_wandb = False
    
    # Load checkpoint if resuming
    start_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.resume and os.path.exists(args.resume):
        start_step, model, optimizer, scaler = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        start_step += 1
        print(f"Resuming from step {start_step}")
    
    # Training loop
    step = start_step
    loss_meter = AverageMeter("loss")
    ppl_meter = AverageMeter("ppl")
    
    print(f"\nStarting training from step {step} to {train_config.max_steps}")
    print(f"Validation every {train_config.val_every} steps")
    if train_config.early_stopping:
        print(f"Early stopping: patience={train_config.early_stopping_patience}")
    
    epoch = 0
    while step < train_config.max_steps:
        epoch += 1
        
        for batch in train_loader:
            if step >= train_config.max_steps:
                break
            
            # Training step
            loss, perplexity, grad_norm = train_step(
                model, batch, model_config, train_config,
                optimizer, scaler, device, alpha, step
            )
            
            # Update meters
            if torch.isfinite(torch.tensor(loss)):
                loss_meter.update(loss)
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
                    'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    'train/lr': current_lr,
                }
                logger.log(metrics, step=step)
                loss_meter.reset()
                ppl_meter.reset()
            
            # Validation
            if step % train_config.val_every == 0 and val_loader is not None:
                print(f"\nValidating at step {step}...")
                val_metrics = evaluate(
                    model, val_loader, model_config, train_config,
                    device, alpha, max_batches=train_config.val_batches
                )
                logger.log(val_metrics, step=step)
                print(f"  Val loss: {val_metrics['val/loss']:.4f}")
                print(f"  Val PPL: {val_metrics['val/perplexity']:.2f}")
                
                # Early stopping check
                if train_config.early_stopping:
                    if val_metrics['val/loss'] < best_val_loss - train_config.early_stopping_min_delta:
                        best_val_loss = val_metrics['val/loss']
                        patience_counter = 0
                        # Save best model
                        best_path = os.path.join(train_config.checkpoint_dir, "checkpoint_best.pt")
                        save_checkpoint(step, model, optimizer, scaler, asdict(train_config), best_path)
                        print(f"  New best model saved!")
                    else:
                        patience_counter += 1
                        print(f"  Patience: {patience_counter}/{train_config.early_stopping_patience}")
                        
                        if patience_counter >= train_config.early_stopping_patience:
                            print(f"\nEarly stopping at step {step}")
                            step = train_config.max_steps
                            break
            
            # Checkpointing
            if step % train_config.save_every == 0:
                checkpoint_path = os.path.join(
                    train_config.checkpoint_dir,
                    f"checkpoint_step_{step:06d}.pt"
                )
                save_checkpoint(step, model, optimizer, scaler, asdict(train_config), checkpoint_path)
            
            step += 1
    
    # Save final checkpoint
    final_path = os.path.join(train_config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(step - 1, model, optimizer, scaler, asdict(train_config), final_path)
    
    logger.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Discrete Diffusion Language Model")
    
    # Config
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model-preset", type=str, default="base",
                       choices=list(MODEL_PRESETS.keys()), help="Model size preset")
    
    # Data
    parser.add_argument("--data-dir", type=str, default=None, help="Path to processed data directory")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to tokenizer")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    
    # Model
    parser.add_argument("--use-rotary-embeddings", action="store_true", help="Use RoPE")
    
    # Checkpointing
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--use-tensorboard", action="store_true", help="Use TensorBoard")
    
    # Test mode
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    # Apply test mode
    if args.test:
        args.model_preset = "tiny"
        args.max_steps = 100
        args.val_every = 50
        args.batch_size = 4
        args.use_rotary_embeddings = True
    
    main(args)
