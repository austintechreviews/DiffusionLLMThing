"""
Sampling and inference for discrete diffusion language model.

Generates text by iteratively denoising from fully masked sequences.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: int,
    T: int,
    mask_token_id: int,
    temperature: float = 1.0,
    unmask_ratio: float = 1.0,
    strategy: str = "confidence",
) -> torch.Tensor:
    """
    Single denoising step: predict tokens at masked positions.
    
    Args:
        model: Diffusion model
        x: Current noisy sequence of shape (batch, seq_len)
        t: Current timestep (0 to T-1)
        T: Total diffusion steps
        mask_token_id: Token ID for [MASK]
        temperature: Sampling temperature
        unmask_ratio: Fraction of masked positions to unmask this step
        strategy: Unmasking strategy ('confidence', 'random', 'all')
    
    Returns:
        Updated sequence with some positions unmasked
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Get model predictions
    t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
    logits = model(x, t_tensor)  # (batch, seq_len, vocab_size)
    
    # Find currently masked positions
    mask_positions = (x == mask_token_id)  # (batch, seq_len)
    
    if mask_positions.sum() == 0:
        return x  # No masked positions to denoise
    
    # Compute probabilities
    probs = F.softmax(logits / temperature, dim=-1)  # (batch, seq_len, vocab_size)
    
    # Sample tokens at all positions
    probs_flat = probs.view(-1, probs.shape[-1])
    samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
    samples = samples.view(batch_size, seq_len)
    
    # Determine which positions to unmask
    if strategy == "all" or unmask_ratio == 1.0:
        # Unmask all at once (faster but potentially lower quality)
        x = samples
    elif strategy == "random":
        # Randomly unmask a fraction
        num_to_unmask = max(1, int(mask_positions.sum().item() * unmask_ratio))
        masked_indices = torch.where(mask_positions.view(-1))[0]
        unmask_indices = masked_indices[torch.randperm(len(masked_indices))[:num_to_unmask]]
        
        x_flat = x.view(-1)
        x_flat[unmask_indices] = samples.view(-1)[unmask_indices]
        x = x_flat.view(batch_size, seq_len)
    else:  # strategy == "confidence"
        # Unmask positions with highest confidence
        confidence = probs.max(dim=-1).values  # (batch, seq_len)
        
        # Get confidence at masked positions
        conf_masked = confidence[mask_positions]
        
        if len(conf_masked) > 0:
            num_to_unmask = max(1, int(len(conf_masked) * unmask_ratio))
            _, top_indices = torch.topk(conf_masked, min(num_to_unmask, len(conf_masked)))
            
            # Create update mask
            update_mask = torch.zeros_like(x, dtype=torch.bool)
            masked_indices = torch.where(mask_positions)[1] if mask_positions.ndim == 2 else torch.where(mask_positions)[0]
            
            # Handle batched case
            if mask_positions.ndim == 2:
                for b in range(batch_size):
                    batch_masked = torch.where(mask_positions[b])[0]
                    if len(batch_masked) > 0:
                        batch_top = min(num_to_unmask // batch_size + 1, len(batch_masked))
                        if len(conf_masked) >= len(batch_masked):
                            update_mask[b, batch_masked[:batch_top]] = True
            else:
                update_mask.view(-1)[masked_indices[top_indices]] = True
            
            x[update_mask] = samples[update_mask]
    
    return x


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    T: int,
    mask_token_id: int,
    batch_size: int = 1,
    seq_len: int = 128,
    temperature: float = 1.0,
    unmask_schedule: str = "linear",
    device: Optional[torch.device] = None,
    progress_callback: callable = None,
) -> torch.Tensor:
    """
    Generate text by iterative denoising from t=T-1 to t=0.
    
    Starts from fully masked sequence and progressively denoises.
    
    Args:
        model: Trained diffusion model
        T: Total diffusion steps
        mask_token_id: Token ID for [MASK]
        batch_size: Number of sequences to generate
        seq_len: Sequence length
        temperature: Sampling temperature
        unmask_schedule: How to schedule unmasking ('linear', 'cosine', 'uniform')
        device: Device to run on
        progress_callback: Optional callback(step, total_steps) for progress tracking
    
    Returns:
        Generated token ids of shape (batch_size, seq_len)
    """
    model.eval()
    device = device or next(model.parameters()).device
    
    # Start from fully masked sequence (t = T)
    x = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    # Compute unmask schedule
    if unmask_schedule == "linear":
        # Linearly increase unmask ratio over time
        unmask_ratios = torch.linspace(0.1, 1.0, T)
    elif unmask_schedule == "cosine":
        # Cosine schedule: more unmasking at later timesteps
        unmask_ratios = torch.cos(torch.linspace(torch.pi/2, 0, T)) ** 2
        unmask_ratios = torch.clamp(unmask_ratios, min=0.1)
    elif unmask_schedule == "uniform":
        # Unmask roughly same amount each step
        unmask_ratios = torch.ones(T) * (seq_len / T) / seq_len
    else:
        unmask_ratios = torch.ones(T)
    
    # Iteratively denoise from t=T-1 to t=0
    for step, t in enumerate(reversed(range(T))):
        # Compute unmask ratio for this step
        unmask_ratio = unmask_ratios[step].item()
        
        # Determine unmask strategy based on timestep
        if t == 0:
            # At final step, unmask everything
            strategy = "all"
            unmask_ratio = 1.0
        elif unmask_ratio >= 0.9:
            strategy = "all"
        else:
            strategy = "confidence"
        
        # Single denoising step
        x = sample_step(
            model=model,
            x=x,
            t=t,
            T=T,
            mask_token_id=mask_token_id,
            temperature=temperature,
            unmask_ratio=unmask_ratio,
            strategy=strategy,
        )
        
        # Progress callback
        if progress_callback is not None:
            progress_callback(step + 1, T)
    
    return x


@torch.no_grad()
def sample_with_classifier_free_guidance(
    model: torch.nn.Module,
    T: int,
    mask_token_id: int,
    batch_size: int = 1,
    seq_len: int = 128,
    temperature: float = 1.0,
    guidance_scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sampling with classifier-free guidance (if model supports it).
    
    For models trained with conditional inputs, this allows trading off
    diversity vs. adherence to conditions.
    
    Args:
        model: Trained diffusion model
        T: Total diffusion steps
        mask_token_id: Token ID for [MASK]
        batch_size: Number of sequences to generate
        seq_len: Sequence length
        temperature: Sampling temperature
        guidance_scale: CFG scale (1.0 = no guidance, >1 = stronger guidance)
        device: Device to run on
    
    Returns:
        Generated token ids of shape (batch_size, seq_len)
    """
    model.eval()
    device = device or next(model.parameters()).device
    
    # Start from fully masked sequence
    x = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    # Iteratively denoise
    for t in reversed(range(T)):
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
        
        # Get model predictions
        logits = model(x, t_tensor)  # (batch, seq_len, vocab_size)
        
        # Apply guidance (for now, just use raw logits)
        # In CFG, you'd compute: guided = unconditional + scale * (conditional - unconditional)
        if guidance_scale != 1.0:
            logits = logits * guidance_scale
        
        # Find masked positions
        mask_positions = (x == mask_token_id)
        
        if mask_positions.sum() == 0:
            continue
        
        # Sample tokens
        probs = F.softmax(logits / temperature, dim=-1)
        probs_flat = probs.view(-1, probs.shape[-1])
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        samples = samples.view(batch_size, seq_len)
        
        # Update masked positions (unmask all at each step for simplicity)
        x = samples
    
    return x


def tokens_to_text(
    tokens: torch.Tensor,
    tokenizer=None,
    eos_token_id: int = 2,
) -> list:
    """
    Convert token ids to text strings.
    
    Args:
        tokens: Token ids of shape (batch, seq_len) or (seq_len,)
        tokenizer: Optional tokenizer with decode method
        eos_token_id: EOS token ID for truncation
    
    Returns:
        List of decoded strings
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    
    batch_size = tokens.shape[0]
    texts = []
    
    for i in range(batch_size):
        seq = tokens[i].tolist()
        
        # Truncate at EOS
        if eos_token_id in seq:
            seq = seq[:seq.index(eos_token_id)]
        
        if tokenizer is not None:
            text = tokenizer.decode(seq, skip_special_tokens=True)
        else:
            # Simple character-level decoding (for testing)
            text = "".join(chr(min(t % 128, 127)) for t in seq if t > 2)
        
        texts.append(text.strip())
    
    return texts
