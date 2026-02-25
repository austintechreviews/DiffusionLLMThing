"""
Diffusion process functions for discrete diffusion language model.

Implements the absorbing/masking diffusion process and loss computation.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def get_noise_schedule(
    T: int,
    schedule: str = "linear",
) -> torch.Tensor:
    """
    Compute the noise schedule (alpha_t values) for diffusion.
    
    Args:
        T: Total number of diffusion steps
        schedule: Schedule type ('linear' or 'cosine')
    
    Returns:
        alpha_t values of shape (T+1,) where alpha_t = P(unchanged at step t)
    """
    t = torch.arange(T + 1)
    
    if schedule == "linear":
        # Linear schedule: alpha_t = 1 - t/T
        alpha = 1.0 - t.float() / T
    elif schedule == "cosine":
        # Cosine schedule (more stable at boundaries)
        s = 0.008
        alpha = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
        alpha = alpha / alpha[0]  # Normalize so alpha_0 = 1
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return alpha


def forward_diffusion(
    x0: torch.Tensor,
    t: torch.Tensor,
    T: int,
    mask_token_id: int,
    pad_token_id: int = -1,
    alpha: torch.Tensor = None,
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
        alpha: Precomputed alpha schedule (optional, uses linear if None)
    
    Returns:
        xt: Noisy token ids of shape (batch, seq_len)
        mask_positions: Boolean mask of shape (batch, seq_len) indicating masked positions
    """
    batch_size, seq_len = x0.shape
    device = x0.device
    
    # Handle scalar t
    if t.ndim == 0:
        t = t.expand(batch_size)
    
    # Compute masking probability from alpha schedule
    if alpha is not None:
        alpha_t = alpha[t].to(device)  # (batch,)
    else:
        # Linear schedule: alpha_t = 1 - t/T, mask_prob = t/T
        alpha_t = 1.0 - t.float() / T
    
    mask_prob = 1.0 - alpha_t  # (batch,)
    mask_prob = mask_prob.unsqueeze(1).expand(batch_size, seq_len)  # (batch, seq_len)
    
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
    alpha: torch.Tensor = None,
    reduction: str = "mean",
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
        alpha: Precomputed alpha schedule (optional)
        reduction: How to reduce loss ('mean' or 'sum')
    
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
    if alpha is not None:
        alpha_t = alpha[t].to(device)  # (batch,)
    else:
        alpha_t = 1.0 - t.float() / T
    
    # Avoid division by zero at t=T (alpha_t=0)
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
    if reduction == "mean":
        num_masked = mask_flat.sum()
        loss = weighted_loss.sum() / num_masked
    else:  # reduction == "sum"
        loss = weighted_loss.sum()
    
    return loss


def compute_transition_prob(
    t: torch.Tensor,
    T: int,
    alpha: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the transition probability q(x_t | x_{t-1}).
    
    For absorbing diffusion:
    - If x_{t-1} is masked, x_t stays masked with probability 1
    - If x_{t-1} is unmasked, x_t becomes masked with probability 1/(T-t+1)
    
    Args:
        t: Timestep (1-indexed, from 1 to T)
        T: Total diffusion steps
        alpha: Precomputed alpha schedule
    
    Returns:
        Transition probability (probability of becoming masked)
    """
    if alpha is not None:
        # q(x_t | x_0) = alpha_t for unmasked
        # q(x_t | x_{t-1}) = alpha_t / alpha_{t-1}
        alpha_t = alpha[t]
        alpha_t_minus_1 = alpha[t - 1]
        prob_mask = 1.0 - alpha_t / (alpha_t_minus_1 + 1e-8)
    else:
        # Linear schedule
        prob_mask = 1.0 / (T - t.float() + 1)
    
    return prob_mask
