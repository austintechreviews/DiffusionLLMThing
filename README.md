# Discrete Diffusion Language Model

A PyTorch implementation of discrete diffusion for language modeling, following the Mercury/MDLM approach. The model operates by corrupting token sequences with a masking noise process and learning to denoise them.

## Overview

This implementation features:
- **Absorbing/masking diffusion**: Tokens are independently masked with probability `t/T` at timestep `t`
- **Transformer denoiser**: AdaLN-based timestep conditioning injected into each transformer layer
- **MDLM ELBO loss**: Cross-entropy over masked positions with proper weighting
- **Efficient training**: Mixed precision, gradient clipping, cosine LR with warmup
- **Complete tooling**: Training, sampling, and evaluation scripts

## Quick Start

```bash
# Clone the repository
git clone https://github.com/austintechreviews/DiffusionLLMThing.git
cd DiffusionLLMThing

# Setup virtual environment and install dependencies
./setup.sh              # Core dependencies only
./setup.sh --dev        # Include development dependencies (pytest, black, ruff)
./setup.sh --all        # Include all optional dependencies (wandb, tensorboard)

# Activate the virtual environment
source venv/bin/activate

# Or use the run helper
./run.sh python scripts/train.py --config configs/default.yaml
```

## Installation

### Using setup script (recommended)

```bash
./setup.sh --dev    # Recommended for development
source venv/bin/activate
```

### Manual installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e ".[all]"
```

## Usage

### Training

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Train with test config (smaller, faster)
python scripts/train.py --config configs/test.yaml

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_step_005000.pt

# Override config from command line
python scripts/train.py --data-path data/my_data.txt --checkpoint-dir my_checkpoints
```

### Sampling

```bash
# Generate samples
python scripts/sample.py --checkpoint checkpoints/checkpoint_final.pt

# Generate more samples with custom parameters
python scripts/sample.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --num-samples 10 \
    --max-length 256 \
    --temperature 0.8 \
    --output samples.txt
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --data-path data/test.txt

# Evaluate at different timesteps
python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --data-path data/test.txt \
    --eval-timesteps
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Diffusion process
T: 1000  # Number of diffusion steps

# Model architecture
vocab_size: 32000
hidden_dim: 512
num_layers: 6
num_heads: 8

# Training
lr: 0.0003
batch_size: 32
seq_len: 128
warmup_steps: 2000
max_steps: 100000
```

## Project Structure

```
DiffusionLLMThing/
├── configs/
│   ├── default.yaml      # Default training config
│   └── test.yaml         # Small config for testing
├── diffusionllm/
│   ├── __init__.py
│   ├── model.py          # Transformer architecture
│   ├── diffusion.py      # Diffusion process functions
│   ├── sampling.py       # Generation/inference
│   └── utils.py          # Checkpointing, logging, etc.
├── scripts/
│   ├── train.py          # Training entry point
│   ├── sample.py         # Sampling entry point
│   └── evaluate.py       # Evaluation entry point
├── tests/
│   ├── test_diffusion.py
│   ├── test_model.py
│   ├── test_sampling.py
│   ├── test_training.py
│   └── test_utils.py
├── setup.sh              # Setup script
├── run.sh                # Run helper script
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# Activate venv first
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_diffusion.py -v

# Run with coverage
pytest tests/ --cov=diffusionllm --cov-report=html
```

## API Usage

```python
import torch
from diffusionllm import (
    DiscreteDiffusionTransformer,
    ModelConfig,
    forward_diffusion,
    sample,
)

# Create model
config = ModelConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
)
model = DiscreteDiffusionTransformer(config)

# Forward diffusion (corrupt data)
x0 = torch.randint(3, 32000, (4, 128))  # Clean tokens
t = torch.randint(0, 1000, (4,))        # Timesteps
xt, mask = forward_diffusion(x0, t, T=1000, mask_token_id=0)

# Denoise
logits = model(xt, t)

# Generate samples
generated = sample(
    model, 
    T=1000, 
    mask_token_id=0,
    batch_size=4, 
    seq_len=128
)
```

## Algorithm Details

### Forward Diffusion

At timestep `t`, each non-padding token is independently masked with probability `t/T`:

```
q(x_t | x_0) = ∏_i Bernoulli(mask_i | t/T)
```

### Loss Function

The MDLM ELBO-style loss weights each timestep by `1/α_t` where `α_t = 1 - t/T`:

```
L = E_t[1/α_t * E_q(x_t|x_0)[CE(model(x_t, t), x_0) over masked positions]]
```

### Sampling

Generation starts from a fully masked sequence and iteratively denoises from `t=T-1` to `t=0`, sampling tokens at masked positions based on model predictions.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA GPU (recommended for training)

## License

MIT License

## References

- [Mercury: Fast Token Generation with Discrete Diffusion](https://arxiv.org/abs/2401.xxxxx)
- [MDLM: Masked Diffusion Language Models](https://arxiv.org/abs/2402.xxxxx)
- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
