"""
Model configuration presets for discrete diffusion language model.

Provides predefined model sizes for different use cases.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the diffusion transformer model."""
    
    # Vocabulary
    vocab_size: int = 32000
    
    # Architecture
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: Optional[int] = None  # Defaults to 4 * hidden_dim
    dropout: float = 0.1
    
    # Sequence
    max_seq_len: int = 512
    
    # Embeddings
    use_rotary_embeddings: bool = False
    rope_theta: float = 10000.0
    
    # Special tokens
    mask_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2
    bos_token_id: int = 3
    
    # Attention
    use_flash_attention: bool = False
    attention_dropout: float = 0.0
    
    # Initialization
    init_std: float = 0.02
    
    @property
    def num_parameters(self) -> int:
        """Estimate number of parameters."""
        # Rough estimate (actual depends on implementation details)
        embed_params = self.vocab_size * self.hidden_dim
        attention_params = 4 * self.hidden_dim * self.hidden_dim * self.num_layers
        ffn_dim = self.dim_feedforward or 4 * self.hidden_dim
        ffn_params = 2 * self.hidden_dim * ffn_dim * self.num_layers
        norm_params = 4 * self.hidden_dim * self.num_layers
        output_params = self.hidden_dim * self.vocab_size
        
        return embed_params + attention_params + ffn_params + norm_params + output_params
    
    @property
    def num_parameters_millions(self) -> float:
        """Get parameter count in millions."""
        return self.num_parameters / 1_000_000


# Predefined model configurations
MODEL_PRESETS = {
    # Micro model for very small datasets/testing
    "micro": ModelConfig(
        vocab_size=8192,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        dropout=0.0,
        use_rotary_embeddings=True,
    ),
    
    # Tiny model for testing/debugging
    "tiny": ModelConfig(
        vocab_size=8192,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=512,
        dropout=0.0,
        use_rotary_embeddings=True,  # Enabled by default
    ),

    # Small model for quick experiments
    "small": ModelConfig(
        vocab_size=16384,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=512,
        dropout=0.1,
        use_rotary_embeddings=True,
    ),

    # Base model (default)
    "base": ModelConfig(
        vocab_size=32000,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        max_seq_len=512,
        dropout=0.1,
        use_rotary_embeddings=True,
    ),

    # Medium model
    "medium": ModelConfig(
        vocab_size=32000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024,
        dropout=0.1,
        use_rotary_embeddings=True,
    ),

    # Large model
    "large": ModelConfig(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=16,
        num_heads=16,
        max_seq_len=1024,
        dropout=0.1,
        use_rotary_embeddings=True,
    ),

    # XL model
    "xl": ModelConfig(
        vocab_size=32000,
        hidden_dim=2048,
        num_layers=24,
        num_heads=16,
        dim_feedforward=8192,
        max_seq_len=2048,
        dropout=0.1,
        use_rotary_embeddings=True,
    ),
}


def get_model_config(preset: str = "base", **overrides) -> ModelConfig:
    """
    Get a model configuration by preset name.
    
    Args:
        preset: Preset name ('tiny', 'small', 'base', 'medium', 'large', 'xl')
        **overrides: Override specific config values
    
    Returns:
        ModelConfig instance
    """
    if preset not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")
    
    # Get base config from preset
    config = MODEL_PRESETS[preset]
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    
    return config


def print_model_summary(config: ModelConfig):
    """Print a summary of the model configuration."""
    print("\n" + "=" * 50)
    print("Model Configuration")
    print("=" * 50)
    print(f"  Vocabulary size:    {config.vocab_size:,}")
    print(f"  Hidden dimension:   {config.hidden_dim}")
    print(f"  Number of layers:   {config.num_layers}")
    print(f"  Number of heads:    {config.num_heads}")
    print(f"  Max sequence len:   {config.max_seq_len}")
    print(f"  Dropout:            {config.dropout}")
    print(f"  Rotary embeddings:  {config.use_rotary_embeddings}")
    print(f"  Flash attention:    {config.use_flash_attention}")
    print("-" * 50)
    print(f"  Estimated params:   {config.num_parameters_millions:.2f}M")
    print("=" * 50 + "\n")
