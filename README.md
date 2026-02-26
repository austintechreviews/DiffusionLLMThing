# Discrete Diffusion Language Model

A production-ready PyTorch implementation of discrete diffusion for language modeling, following the Mercury/MDLM approach.

## Features

- **BPE Tokenization**: Hugging Face tokenizers integration for proper subword tokenization
- **Model Presets**: Predefined configurations (tiny, small, base, medium, large, xl)
- **Rotary Embeddings**: Optional RoPE for better length extrapolation
- **Validation & Early Stopping**: Automatic validation with perplexity tracking
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Mixed Precision**: AMP support for faster training on GPU
- **Experiment Tracking**: WandB and TensorBoard integration
- **REST API**: FastAPI server for deployment
- **Interactive CLI**: Menu-driven setup and execution

## Quick Start

```bash
# Clone and setup
git clone https://github.com/austintechreviews/DiffusionLLMThing.git
cd DiffusionLLMThing

# Interactive setup (recommended)
./setup.sh -i

# Or quick setup
./setup.sh --dev
source venv/bin/activate

# Download a dataset
python scripts/download_data.py --dataset shakespeare --output data/raw/shakespeare

# Prepare the data
python scripts/prepare_data.py --input data/raw/shakespeare --output data/processed

# Train a model
python scripts/train.py --data-dir data/processed --model-preset tiny

# Or use the interactive menu
./run.sh
```

## Installation

### Interactive Setup (Recommended)

```bash
./setup.sh -i    # Interactive wizard with menus
```

The wizard will guide you through:
1. Python version selection
2. Installation type (core/dev/full)
3. Virtual environment location
4. Confirmation and installation
5. Optional test training

### Quick Setup

```bash
./setup.sh --dev    # Development installation
./setup.sh --all    # Full installation (includes wandb, fastapi)
./setup.sh          # Core only
```

### Manual Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 7.0+ GPU (recommended for training)

## Interactive Run Menu

The `./run.sh` script provides an interactive menu:

```bash
./run.sh
```

```
========================================
  Discrete Diffusion LM - Run Menu
========================================

Training:
  1) Quick test (tiny model, 100 steps)
  2) Train with preset (tiny/small/base)
  3) Resume from checkpoint

Generation:
  4) Interactive chat
  5) Batch sampling
  6) Start API server

Data:
  7) Prepare data
  8) Evaluate model

System:
  9) Check installation
  10) Run tests
  11) Python REPL

  0) Exit
```

### Direct Commands

```bash
# Training
./run.sh train --test           # Quick test (tiny, 100 steps)
./run.sh train                  # Guided training setup
./run.sh resume                 # Resume from checkpoint

# Generation
./run.sh chat                   # Interactive chat
./run.sh sample                 # Batch sampling
./run.sh server --port 8000     # API server

# Data
./run.sh prepare                # Prepare data
./run.sh eval                   # Evaluate model

# System
./run.sh check                  # Check installation
./run.sh test                   # Run test suite
./run.sh python script.py       # Run Python script
```

## Data Preparation

### Download Datasets

Quickly get started with pre-packaged datasets:

```bash
# List available datasets
python scripts/download_data.py --list

# Download Tiny Shakespeare (1MB, good for testing)
python scripts/download_data.py --dataset shakespeare --output data/raw/shakespeare

# Download WikiText-2 (2MB, Wikipedia articles)
python scripts/download_data.py --dataset wikitext-2 --output data/raw/wikitext2

# Download enwik8 (100MB, Wikipedia XML dump)
python scripts/download_data.py --dataset enwik8 --output data/raw/enwik8

# Download from custom URL
python scripts/download_data.py --url https://example.com/data.txt --output data/raw/custom
```

**Available datasets:**

| Dataset | Size | Description |
|---------|------|-------------|
| shakespeare | ~1MB | Tiny Shakespeare - character-level text |
| wikitext-2 | ~2MB | Wikipedia articles |
| wikitext-103 | ~600MB | Larger Wikipedia corpus |
| enwik8 | ~100MB | First 100MB of Wikipedia XML |
| enwik9 | ~1GB | First 1GB of Wikipedia XML |

### Format your data

Place your text files in a directory:

```
data/raw_text/
├── book1.txt
├── book2.txt
├── articles.jsonl
└── ...
```

Supported formats:
- `.txt` - Plain text files
- `.json` - JSON with "text" field
- `.jsonl` - JSON lines with "text" field

### Prepare the data

```bash
# Using run.sh (interactive)
./run.sh prepare

# Or directly
python scripts/prepare_data.py \
    --input data/raw_text/ \
    --output data/processed \
    --vocab-size 32000 \
    --min-frequency 2
```

This will:
1. Train a BPE tokenizer on your data
2. Split into train/val/test (90/5/5)
3. Tokenize all splits
4. Save metadata

Output:
```
data/processed/
├── tokenizer.json      # BPE tokenizer
├── train.jsonl         # Tokenized training data
├── val.jsonl           # Tokenized validation data
├── test.jsonl          # Tokenized test data
└── metadata.json       # Dataset statistics
```

## Training

### Quick Test

```bash
./run.sh train --test
# or
python scripts/train.py --test
```

Runs a tiny model (~5M params) for 100 steps - completes in ~1 minute on CPU.

### Full Training

```bash
# Using run.sh (guided)
./run.sh train

# Or directly
python scripts/train.py \
    --data-dir data/processed \
    --model-preset base \
    --batch-size 32 \
    --max-steps 100000
```

### Model Presets

| Preset  | Params | Hidden | Layers | Heads | Max Len | Use Case |
|---------|--------|--------|--------|-------|---------|----------|
| tiny    | ~5M    | 128    | 2      | 4     | 256     | Testing |
| small   | ~20M   | 256    | 4      | 8     | 512     | Quick experiments |
| base    | ~85M   | 512    | 6      | 8     | 512     | Standard |
| medium  | ~200M  | 768    | 12     | 12    | 1024    | Production |
| large   | ~400M  | 1024   | 16     | 16    | 1024    | High quality |
| xl      | ~1.5B  | 2048   | 24     | 16    | 2048    | Best quality |

### Advanced Training Options

```bash
python scripts/train.py \
    --data-dir data/processed \
    --model-preset base \
    --batch-size 16 \
    --grad-accum 4 \              # Effective batch: 16*4=64
    --lr 3e-4 \
    --warmup-steps 2000 \
    --max-steps 100000 \
    --val-every 1000 \
    --early-stopping \
    --early-stopping-patience 5 \
    --use-rotary-embeddings \
    --use-wandb \
    --checkpoint-dir checkpoints/my_experiment
```

### Resuming Training

```bash
./run.sh resume
# or
python scripts/train.py --resume checkpoints/checkpoint_step_005000.pt
```

## Generation

### Interactive Chat

```bash
./run.sh chat
# or
python scripts/chat.py --checkpoint checkpoints/checkpoint_final.pt
```

**Chat commands:**
- `/quit` - Exit
- `/help` - Show help
- `/temp <0.1-2.0>` - Set temperature
- `/len <32-512>` - Set max length
- `/progress` - Toggle progress bar
- `/model` - Show model info
- `/clear` - Clear history

### Batch Sampling

```bash
./run.sh sample
# or
python scripts/sample.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --num-samples 10 \
    --max-length 256 \
    --temperature 0.8 \
    --output samples.txt
```

### API Server

```bash
./run.sh server --port 8000
# or
python scripts/server.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --port 8000 \
    --host 0.0.0.0
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model` | GET | Model information |
| `/generate` | POST | Generate text |
| `/load` | POST | Load checkpoint |

**Example request:**

```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Once upon a time",
        "max_length": 128,
        "temperature": 0.8,
        "num_sequences": 3
    }'
```

**Python client:**

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "The future of AI",
        "max_length": 128,
        "temperature": 0.8,
        "num_sequences": 3,
    }
)

print(response.json()["generations"])
```

## Evaluation

```bash
./run.sh eval
# or
python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --data-path data/processed/test.jsonl \
    --eval-timesteps
```

Outputs:
- Validation loss
- Perplexity
- Performance at different noise levels

## Configuration

### YAML Config File

```yaml
# configs/my_config.yaml
model_preset: base
data_dir: data/processed
batch_size: 32
lr: 0.0003
max_steps: 100000
warmup_steps: 2000
T: 1000
use_rotary_embeddings: true
use_amp: true
use_wandb: true
```

```bash
python scripts/train.py --config configs/my_config.yaml
```

## Project Structure

```
DiffusionLLMThing/
├── diffusionllm/           # Core package
│   ├── __init__.py
│   ├── config.py          # Model presets
│   ├── model.py           # Transformer architecture
│   ├── diffusion.py       # Diffusion functions
│   ├── sampling.py        # Generation
│   ├── tokenizer.py       # BPE tokenizer wrapper
│   ├── data.py            # Datasets
│   └── utils.py           # Utilities
├── scripts/
│   ├── prepare_data.py    # Data preprocessing
│   ├── train.py           # Training
│   ├── sample.py          # Batch sampling
│   ├── chat.py            # Interactive chat
│   └── server.py          # FastAPI server
├── configs/
│   ├── default.yaml
│   └── test.yaml
├── tests/
├── setup.sh               # Interactive setup
├── run.sh                 # Interactive runner
├── requirements.txt
└── README.md
```

## API Usage

```python
from diffusionllm import (
    get_model_config,
    DiscreteDiffusionTransformer,
    DiffusionTokenizer,
    forward_diffusion,
    sample,
    load_datasets,
)

# Get model config from preset
config = get_model_config("base")
print(f"Parameters: {config.num_parameters_millions:.1f}M")

# Create model
model = DiscreteDiffusionTransformer(config)

# Load tokenizer (if available)
tokenizer = DiffusionTokenizer.load("data/processed/tokenizer.json")

# Encode text
ids = tokenizer.encode("Hello, world!", add_bos=True, add_eos=True)

# Load datasets
dataloaders = load_datasets(
    "data/processed",
    max_length=512,
    batch_size=32,
)

# Generate
generated = sample(
    model=model,
    T=1000,
    mask_token_id=tokenizer.mask_token_id,
    batch_size=1,
    seq_len=128,
    temperature=0.8,
)

# Decode
text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
```

## Algorithm

### Forward Diffusion

At timestep `t`, each token is masked with probability `t/T`:
```
q(x_t | x_0) = ∏ Bernoulli(masked | t/T)
```

### Loss Function

ELBO-weighted cross-entropy:
```
L = E_t[1/α_t × CE(model(x_t, t), x_0)]
```
where `α_t = 1 - t/T`

### Sampling

Iterative denoising from `t=T-1` to `t=0`, progressively unmasking tokens.

## Troubleshooting

### CUDA incompatible with GPU

If your GPU isn't supported by the installed PyTorch CUDA version:
```bash
# The scripts will automatically fall back to CPU
# For faster CPU training, use smaller models:
./run.sh train --test  # Tiny model
```

### Out of memory

Reduce batch size or use gradient accumulation:
```bash
python scripts/train.py \
    --batch-size 8 \
    --grad-accum 4  # Effective batch: 32
```

### Missing optional dependencies

The package works without optional dependencies. Install as needed:
```bash
pip install tokenizers      # For BPE tokenization
pip install fastapi uvicorn # For API server
pip install wandb           # For experiment tracking
```

## License

MIT License

## References

- [Mercury: Fast Token Generation with Discrete Diffusion](https://arxiv.org/abs/2401.xxxxx)
- [MDLM: Masked Diffusion Language Models](https://arxiv.org/abs/2402.xxxxx)
- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
