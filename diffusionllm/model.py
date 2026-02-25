"""
Model architecture for discrete diffusion language model.

Transformer-based denoiser with AdaLN timestep conditioning.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the diffusion transformer model."""
    
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    
    # Special tokens
    mask_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2


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
