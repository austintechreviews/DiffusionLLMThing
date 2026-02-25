"""
Discrete Diffusion Language Model Training Loop (Mercury/MDLM style)

Implements an absorbing/masking diffusion process over token sequences,
where the model learns to denoise by predicting original tokens at masked positions.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Diffusion process
    T: int = 1000  # Total diffusion timesteps
    
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
    
    # Special tokens
    mask_token_id: int = 0  # [MASK] token ID
    pad_token_id: int = 1   # [PAD] token ID
    eos_token_id: int = 2   # [EOS] token ID


# =============================================================================
# Model Components
# =============================================================================

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embeddings followed by an MLP.
    Projects timestep t to a vector of hidden_dim.
    """
    
    def __init__(self, hidden_dim: int, max_period: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_period = max_period
        
        # MLP to project sinusoidal embeddings
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (batch,) with timesteps
        Returns:
            Tensor of shape (batch, hidden_dim)
        """
        half_dim = self.hidden_dim // 2
        device = t.device
        
        # Sinusoidal embeddings
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)   # (batch, hidden_dim)
        
        # MLP projection
        emb = self.mlp(emb)
        return emb


class AdaLN(nn.Module):
    """
    Adaptive LayerNorm: LayerNorm modulated by timestep embeddings.
    Used to inject timestep information into transformer layers.
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.ada_lin = nn.Linear(hidden_dim, hidden_dim * 2)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, hidden_dim)
            t_emb: Tensor of shape (batch, hidden_dim) with timestep embeddings
        Returns:
            Tensor of shape (batch, seq_len, hidden_dim)
        """
        # Project timestep to scale and shift parameters
        shift, scale = self.ada_lin(t_emb).chunk(2, dim=-1)
        shift = shift.unsqueeze(1)  # (batch, 1, hidden_dim)
        scale = scale.unsqueeze(1)
        
        # Apply adaptive normalization
        x = self.norm(x)
        return x * (1 + scale) + shift


class DiscreteDiffusionTransformer(nn.Module):
    """
    Transformer-based denoiser for discrete diffusion.
    
    Takes noisy token sequences and timestep, outputs logits for denoising.
    Uses AdaLN to inject timestep information into each transformer layer.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token embeddings (includes mask token at index 0)
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        
        # Timestep embeddings
        self.timestep_embed = TimestepEmbedding(config.hidden_dim)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_dim))
        
        # Transformer encoder layers with AdaLN
        encoder_layers = []
        for _ in range(config.num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # Wrap with AdaLN
            encoder_layers.append(TransformerBlockWithAdaLN(layer, config.hidden_dim))
        
        self.encoder = nn.ModuleList(encoder_layers)
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Output projection to vocabulary
        self.out_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for embeddings, xavier for others."""
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy token ids of shape (batch, seq_len)
            t: Timestep of shape (batch,) or scalar
        Returns:
            Logits over vocabulary of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token embeddings
        h = self.token_embed(x)  # (batch, seq_len, hidden_dim)
        
        # Add position embeddings (truncate if needed)
        h = h + self.pos_embed[:, :seq_len, :]
        
        # Timestep embeddings
        if t.ndim == 0:
            t = t.expand(batch_size)
        t_emb = self.timestep_embed(t)  # (batch, hidden_dim)
        
        # Pass through transformer layers with AdaLN
        for layer in self.encoder:
            h = layer(h, t_emb)
        
        # Final normalization and output projection
        h = self.final_norm(h)
        logits = self.out_proj(h)  # (batch, seq_len, vocab_size)
        
        return logits


class TransformerBlockWithAdaLN(nn.Module):
    """
    Transformer encoder block with Adaptive LayerNorm for timestep conditioning.
    """
    
    def __init__(self, layer: nn.TransformerEncoderLayer, hidden_dim: int):
        super().__init__()
        self.layer = layer
        self.ada_ln1 = AdaLN(hidden_dim)
        self.ada_ln2 = AdaLN(hidden_dim)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Self-attention block with AdaLN
        h = self.ada_ln1(x, t_emb)
        h = self.layer.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.layer.dropout1(h)
        
        # Feed-forward block with AdaLN
        h = self.ada_ln2(x, t_emb)
        h = self.layer.linear2(self.layer.dropout(self.layer.activation(self.layer.linear1(h))))
        x = x + self.layer.dropout2(h)
        
        return x


# =============================================================================
# Diffusion Process
# =============================================================================

def forward_diffusion(
    x0: torch.Tensor,
    t: torch.Tensor,
    T: int,
    mask_token_id: int,
    pad_token_id: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion process: corrupt x_0 by masking tokens.
    
    At timestep t, each non-pad token is independently masked with probability t/T.
    This implements the absorbing state diffusion where [MASK] is the absorbing state.
    
    Args:
        x0: Clean token ids of shape (batch, seq_len)
        t: Timestep of shape (batch,) or scalar (0 to T-1)
        T: Total number of diffusion steps
        mask_token_id: Token ID for [MASK]
        pad_token_id: Token ID for padding (these are never masked)
    
    Returns:
        xt: Noisy token ids of shape (batch, seq_len)
        mask_positions: Boolean mask of shape (batch, seq_len) indicating masked positions
    """
    batch_size, seq_len = x0.shape
    device = x0.device
    
    # Handle scalar t
    if t.ndim == 0:
        t = t.expand(batch_size)
    
    # Compute masking probability: alpha_t = 1 - t/T
    # Probability of being masked = t/T
    t_normalized = t.float() / T  # (batch,)
    mask_prob = t_normalized.unsqueeze(1).expand(batch_size, seq_len)  # (batch, seq_len)
    
    # Sample which positions to mask
    # Only mask non-padding tokens
    is_not_pad = (x0 != pad_token_id) if pad_token_id >= 0 else torch.ones_like(x0, dtype=torch.bool)
    random_uniform = torch.rand_like(x0, dtype=torch.float32)
    mask_positions = (random_uniform < mask_prob) & is_not_pad
    
    # Create noisy sequence by replacing masked positions with mask token
    xt = x0.clone()
    xt[mask_positions] = mask_token_id
    
    return xt, mask_positions


def compute_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    mask_positions: torch.Tensor,
    t: torch.Tensor,
    T: int,
    pad_token_id: int = -1,
) -> torch.Tensor:
    """
    Compute the diffusion loss (MDLM ELBO-style).
    
    Cross-entropy loss over masked positions only, weighted by 1/(1 - t/T).
    This weighting comes from the ELBO for discrete diffusion.
    
    Args:
        logits: Model output of shape (batch, seq_len, vocab_size)
        x0: Original clean tokens of shape (batch, seq_len)
        mask_positions: Boolean mask of shape (batch, seq_len) for masked positions
        t: Timestep of shape (batch,)
        T: Total diffusion steps
        pad_token_id: Padding token ID to ignore
    
    Returns:
        Scalar loss tensor
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    
    # Handle scalar t
    if t.ndim == 0:
        t = t.expand(batch_size)
    
    # Compute MDLM ELBO weight: 1 / (1 - t/T) = 1 / alpha_t
    # This upweights earlier timesteps where fewer tokens are masked
    t_normalized = t.float() / T
    alpha_t = 1 - t_normalized  # (batch,)
    
    # Avoid division by zero at t=T (alpha_t=0)
    # At t=T, all tokens are masked, so we use a small epsilon
    eps = 1e-6
    weights = 1.0 / (alpha_t + eps)  # (batch,)
    
    # Compute cross-entropy loss at each position
    logits_flat = logits.view(-1, vocab_size)
    x0_flat = x0.view(-1)
    mask_flat = mask_positions.view(-1)
    
    # Ignore padding positions
    if pad_token_id >= 0:
        pad_mask = (x0_flat != pad_token_id)
        mask_flat = mask_flat & pad_mask
    
    # Only compute loss at masked positions
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    loss_per_token = F.cross_entropy(logits_flat, x0_flat, reduction='none')  # (batch * seq_len,)
    
    # Apply mask and weights
    mask_flat = mask_flat.float()
    
    # Weight each sample by its timestep weight
    weights_expanded = weights.unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
    weighted_loss = loss_per_token * mask_flat * weights_expanded
    
    # Normalize by number of masked tokens (not total tokens)
    num_masked = mask_flat.sum()
    loss = weighted_loss.sum() / num_masked
    
    return loss


# =============================================================================
# Training Functions
# =============================================================================

def train_step(
    model: nn.Module,
    x0: torch.Tensor,
    config: Config,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Single training step with mixed precision.
    
    Args:
        model: The diffusion model
        x0: Clean token batch of shape (batch, seq_len)
        config: Training configuration
        optimizer: AdamW optimizer
        scaler: AMP GradScaler
        device: Device to run on
    
    Returns:
        loss: Scalar loss
        perplexity: Perplexity on masked tokens
        grad_norm: Gradient norm after clipping
    """
    model.train()
    batch_size = x0.shape[0]
    
    # Sample random timestep for each sample in batch
    t = torch.randint(0, config.T, (batch_size,), device=device)
    
    # Forward diffusion: corrupt x0 to get xt
    xt, mask_positions = forward_diffusion(
        x0, t, config.T, config.mask_token_id, config.pad_token_id
    )
    
    # Forward pass with mixed precision
    optimizer.zero_grad()
    
    with autocast('cuda'):
        logits = model(xt, t)
        loss = compute_loss(
            logits, x0, mask_positions, t, config.T, config.pad_token_id
        )
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # Compute perplexity on masked positions (for logging)
    with torch.no_grad():
        with autocast('cuda'):
            logits_masked = logits[mask_positions]
            x0_masked = x0[mask_positions]
            if len(x0_masked) > 0:
                ce_loss = F.cross_entropy(logits_masked, x0_masked)
                perplexity = torch.exp(ce_loss)
            else:
                perplexity = torch.tensor(float('inf'), device=device)
    
    return loss, perplexity, grad_norm.item()


def get_lr_schedule(step: int, config: Config) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    Args:
        step: Current training step
        config: Configuration with warmup_steps and lr
    
    Returns:
        Learning rate multiplier (multiply by base lr)
    """
    if step < config.warmup_steps:
        # Linear warmup
        return step / config.warmup_steps
    else:
        # Cosine decay
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# Dataset
# =============================================================================

class TextDataset(Dataset):
    """
    Simple text dataset that returns tokenized sequences.
    
    In practice, replace this with your actual dataset loading logic.
    """
    
    def __init__(self, data_path: str, seq_len: int, vocab_size: int):
        """
        Args:
            data_path: Path to text file (one document per line)
            seq_len: Fixed sequence length
            vocab_size: Vocabulary size
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Load and tokenize data
        # For demonstration: create synthetic data
        # Replace with actual tokenization logic
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> torch.Tensor:
        """Load and tokenize text data."""
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                lines = f.readlines()
            
            # Simple tokenization: map characters to IDs (for demo)
            # Replace with proper tokenizer (e.g., BPE, SentencePiece)
            all_tokens = []
            for line in lines:
                tokens = [min(ord(c) % (self.vocab_size - 3) + 3, self.vocab_size - 1) 
                         for c in line.strip()]
                all_tokens.extend(tokens)
                all_tokens.append(2)  # EOS token
            
            # Create fixed-length sequences
            data = []
            for i in range(0, len(all_tokens) - self.seq_len, self.seq_len):
                data.append(all_tokens[i:i + self.seq_len])
            
            return torch.tensor(data, dtype=torch.long)
        else:
            # Synthetic data for testing
            print(f"Data file not found: {data_path}. Using synthetic data.")
            return torch.randint(3, self.vocab_size, (1000, self.seq_len))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# =============================================================================
# Sampling / Inference
# =============================================================================

@torch.no_grad()
def sample(
    model: nn.Module,
    config: Config,
    batch_size: int = 1,
    seq_len: Optional[int] = None,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate text by iterative denoising from t=T to t=0.
    
    Starts from fully masked sequence and progressively denoises.
    At each step, samples tokens at masked positions from the model's prediction.
    
    Args:
        model: Trained diffusion model
        config: Configuration
        batch_size: Number of sequences to generate
        seq_len: Sequence length (defaults to config.seq_len)
        temperature: Sampling temperature
        device: Device to run on
    
    Returns:
        Generated token ids of shape (batch_size, seq_len)
    """
    model.eval()
    seq_len = seq_len or config.seq_len
    device = device or next(model.parameters()).device
    
    # Start from fully masked sequence (t = T)
    x = torch.full((batch_size, seq_len), config.mask_token_id, dtype=torch.long, device=device)
    
    # Iteratively denoise from t=T-1 to t=0
    for t in reversed(range(config.T)):
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
        
        # Get model predictions
        logits = model(x, t_tensor)  # (batch, seq_len, vocab_size)
        
        # Find currently masked positions
        mask_positions = (x == config.mask_token_id)
        
        if mask_positions.sum() == 0:
            break  # No more masked positions
        
        # Sample tokens at masked positions
        probs = F.softmax(logits / temperature, dim=-1)  # (batch, seq_len, vocab_size)
        
        # Sample from categorical distribution
        probs_flat = probs.view(-1, config.vocab_size)
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        samples = samples.view(batch_size, seq_len)
        
        # Update only masked positions
        # Strategy: at each step, unmask some fraction of positions
        # Simple approach: unmask all positions at once at the end
        # Better approach: progressively unmask (like DDPM sampling)
        
        # For simplicity: sample all masked positions at each step
        # and keep the sample with highest confidence
        if t == 0:
            # At final step, commit to all predictions
            x = samples
        else:
            # Progressive decoding: unmask a fraction of positions
            # Compute confidence (max probability) at each position
            confidence = probs.max(dim=-1).values  # (batch, seq_len)
            
            # Determine how many positions to unmask at this step
            # Linear schedule: unmask roughly seq_len/T positions per step
            positions_to_unmask = max(1, seq_len // config.T)
            
            # Select positions with highest confidence among masked
            conf_masked = confidence[mask_positions]
            if len(conf_masked) > 0:
                top_k = min(positions_to_unmask, len(conf_masked))
                _, top_indices = torch.topk(conf_masked, top_k)
                
                # Create update mask
                update_mask = torch.zeros_like(x, dtype=torch.bool)
                masked_indices = torch.where(mask_positions)[1]
                update_mask[:, masked_indices[top_indices]] = True
                
                # Update selected positions
                x[update_mask] = samples[update_mask]
    
    return x


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Config,
    path: str,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, nn.Module, Optional[torch.optim.Optimizer], Optional[GradScaler]]:
    """
    Load training checkpoint.
    
    Returns:
        step: Training step to resume from
        model: Model with loaded weights
        optimizer: Optimizer with loaded state (if provided)
        scaler: Scaler with loaded state (if provided)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Restore RNG state for reproducibility
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    
    return checkpoint['step'], model, optimizer, scaler


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: Config, resume_from: Optional[str] = None):
    """
    Main training loop for discrete diffusion language model.
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize model
    model = DiscreteDiffusionTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda')
    
    # Load checkpoint if resuming
    start_step = 0
    if resume_from is not None and os.path.exists(resume_from):
        start_step, model, optimizer, scaler = load_checkpoint(
            resume_from, model, optimizer, scaler, device
        )
        start_step += 1  # Resume from next step
    
    # Setup data
    # Replace with your actual dataset
    dataset = TextDataset(
        data_path="data/train.txt",
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Training loop
    step = start_step
    while step < config.max_steps:
        for batch in dataloader:
            if step >= config.max_steps:
                break
            
            x0 = batch.to(device)  # (batch, seq_len)
            
            # Training step
            loss, perplexity, grad_norm = train_step(
                model, x0, config, optimizer, scaler, device
            )
            
            # Update learning rate
            lr_mult = get_lr_schedule(step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr * lr_mult
            
            # Logging
            if step % config.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Step {step:6d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"PPL: {perplexity.item():.2f} | "
                    f"Grad Norm: {grad_norm:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Checkpointing
            if step % config.save_every == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir, f"checkpoint_step_{step:06d}.pt"
                )
                save_checkpoint(step, model, optimizer, scaler, config, checkpoint_path)
            
            step += 1
    
    # Save final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(step - 1, model, optimizer, scaler, config, final_path)
    print("Training complete!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Discrete Diffusion Language Model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    args = parser.parse_args()
    
    # Create config (optionally load from file)
    config = Config()
    
    # Override with command line args if provided
    # (In practice, you might want to use a config file parser like OmegaConf)
    
    # Start training
    train(config, resume_from=args.resume)
