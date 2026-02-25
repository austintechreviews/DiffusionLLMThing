"""
Model architecture for discrete diffusion language model.

Transformer-based denoiser with AdaLN timestep conditioning.
Supports rotary embeddings and flash attention.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


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


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Applies rotation matrices to query and key based on position.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build rotation cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        """Precompute rotation matrices."""
        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for positions.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, heads, seq_len, dim)
            position_ids: Optional position IDs (defaults to 0..seq_len-1)
        
        Returns:
            (cos, sin) tensors for rotation
        """
        if position_ids is not None:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            seq_len = x.shape[-2] if x.dim() == 4 else x.shape[1]
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    # Expand cos/sin to match q/k shape
    if q.dim() == 4:  # (batch, heads, seq_len, dim)
        cos = cos.unsqueeze(1)  # (1, 1, seq_len, dim/2)
        sin = sin.unsqueeze(1)
    
    cos = cos.unsqueeze(-1)  # (..., seq_len, dim/2, 1)
    sin = sin.unsqueeze(-1)
    
    # Duplicate cos/sin for the two halves
    cos = torch.cat([cos, cos], dim=-1).flatten(-2)
    sin = torch.cat([sin, sin], dim=-1).flatten(-2)
    
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot


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


class TransformerBlockWithAdaLN(nn.Module):
    """
    Transformer encoder block with Adaptive LayerNorm for timestep conditioning.
    
    Supports rotary embeddings and optional flash attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        use_rotary_embeddings: bool = False,
        rope_theta: float = 10000.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        
        # AdaLN for conditioning
        self.ada_ln1 = AdaLN(hidden_dim)
        self.ada_ln2 = AdaLN(hidden_dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        # Rotary embeddings
        if use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(hidden_dim // num_heads, theta=rope_theta)
        
        # Feed-forward
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()
        
        # Flash attention (if available)
        self.use_flash_attention = use_flash_attention
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention block with AdaLN
        h = self.ada_ln1(x, t_emb)
        
        if self.use_rotary_embeddings:
            # Apply RoPE
            batch_size, seq_len, _ = h.shape
            h = h.view(batch_size, seq_len, self.num_heads, self.hidden_dim // self.num_heads)
            h = h.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
            
            cos, sin = self.rotary_emb(h)
            h, _ = apply_rotary_pos_emb(h, h, cos, sin)
            
            h = h.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        # Attention
        h = self.self_attn(h, h, h, attn_mask=attention_mask, need_weights=False)[0]
        x = x + self.dropout1(h)
        
        # Feed-forward block with AdaLN
        h = self.ada_ln2(x, t_emb)
        h = self.linear2(self.dropout(self.activation(self.linear1(h))))
        x = x + self.dropout2(h)
        
        return x


class DiscreteDiffusionTransformer(nn.Module):
    """
    Transformer-based denoiser for discrete diffusion.
    
    Takes noisy token sequences and timestep, outputs logits for denoising.
    Uses AdaLN to inject timestep information into each transformer layer.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (includes mask token at index 0)
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        
        # Timestep embeddings
        self.timestep_embed = TimestepEmbedding(config.hidden_dim)
        
        # Position embeddings (only if not using RoPE)
        if not config.use_rotary_embeddings:
            self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_dim))
        else:
            self.register_buffer("pos_embed", torch.zeros(1, 1, 1))  # Dummy
        
        # Transformer encoder layers with AdaLN
        dim_ffn = config.dim_feedforward or 4 * config.hidden_dim
        
        self.encoder = nn.ModuleList([
            TransformerBlockWithAdaLN(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dim_feedforward=dim_ffn,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                use_rotary_embeddings=config.use_rotary_embeddings,
                rope_theta=config.rope_theta,
                use_flash_attention=config.use_flash_attention,
            )
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Output projection to vocabulary
        self.out_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for embeddings, xavier for others."""
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.config.init_std)
        
        if not self.config.use_rotary_embeddings:
            nn.init.normal_(self.pos_embed, mean=0.0, std=self.config.init_std)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy token ids of shape (batch, seq_len)
            t: Timestep of shape (batch,) or scalar
            attention_mask: Optional attention mask of shape (batch, seq_len)
        Returns:
            Logits over vocabulary of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token embeddings
        h = self.token_embed(x)  # (batch, seq_len, hidden_dim)
        
        # Add position embeddings (if not using RoPE)
        if not self.config.use_rotary_embeddings:
            h = h + self.pos_embed[:, :seq_len, :]
        
        # Timestep embeddings
        if t.ndim == 0:
            t = t.expand(batch_size)
        t_emb = self.timestep_embed(t)  # (batch, hidden_dim)
        
        # Create attention mask (convert to additive mask)
        attn_mask = None
        if attention_mask is not None:
            # (batch, seq_len) -> (batch, 1, 1, seq_len) for MultiheadAttention
            attn_mask = (1.0 - attention_mask.float()) * -1e9
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        
        # Pass through transformer layers with AdaLN
        for layer in self.encoder:
            h = layer(h, t_emb, attention_mask=attn_mask)
        
        # Final normalization and output projection
        h = self.final_norm(h)
        logits = self.out_proj(h)  # (batch, seq_len, vocab_size)
        
        return logits
