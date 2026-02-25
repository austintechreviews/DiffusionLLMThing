"""
Discrete Diffusion Language Model (Mercury/MDLM style)

A transformer-based model that learns to denoise token sequences
corrupted by a masking diffusion process.
"""

from .model import DiscreteDiffusionTransformer, TimestepEmbedding, AdaLN
from .diffusion import forward_diffusion, compute_loss, get_noise_schedule
from .sampling import sample, sample_step
from .utils import (
    save_checkpoint,
    load_checkpoint,
    get_lr_schedule,
    count_parameters,
)

__version__ = "0.1.0"
__all__ = [
    "DiscreteDiffusionTransformer",
    "TimestepEmbedding",
    "AdaLN",
    "forward_diffusion",
    "compute_loss",
    "get_noise_schedule",
    "sample",
    "sample_step",
    "save_checkpoint",
    "load_checkpoint",
    "get_lr_schedule",
    "count_parameters",
]
