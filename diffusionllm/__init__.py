"""
Discrete Diffusion Language Model (Mercury/MDLM style)

A transformer-based model that learns to denoise token sequences
corrupted by a masking diffusion process.
"""

from .config import ModelConfig, MODEL_PRESETS, get_model_config, print_model_summary
from .model import DiscreteDiffusionTransformer, TimestepEmbedding, AdaLN
from .diffusion import forward_diffusion, compute_loss, get_noise_schedule
from .sampling import sample, sample_step, tokens_to_text
from .utils import (
    save_checkpoint,
    load_checkpoint,
    get_lr_schedule,
    count_parameters,
    AverageMeter,
    TrainingLogger,
)

# Optional imports (require additional dependencies)
try:
    from .tokenizer import DiffusionTokenizer
except ImportError:
    DiffusionTokenizer = None  # type: ignore

try:
    from .data import TokenizedDataset, StreamingDataset, create_dataloader, load_datasets
except ImportError:
    TokenizedDataset = None  # type: ignore
    StreamingDataset = None  # type: ignore
    create_dataloader = None  # type: ignore
    load_datasets = None  # type: ignore

__version__ = "0.2.0"
__all__ = [
    # Config
    "ModelConfig",
    "MODEL_PRESETS",
    "get_model_config",
    "print_model_summary",
    # Model
    "DiscreteDiffusionTransformer",
    "TimestepEmbedding",
    "AdaLN",
    # Diffusion
    "forward_diffusion",
    "compute_loss",
    "get_noise_schedule",
    # Sampling
    "sample",
    "sample_step",
    "tokens_to_text",
    # Utils
    "save_checkpoint",
    "load_checkpoint",
    "get_lr_schedule",
    "count_parameters",
    "AverageMeter",
    "TrainingLogger",
    # Tokenizer (optional)
    "DiffusionTokenizer",
    # Data (optional)
    "TokenizedDataset",
    "StreamingDataset",
    "create_dataloader",
    "load_datasets",
]
