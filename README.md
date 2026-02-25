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
- **Production Scripts**: Data preparation, training, sampling, chat, and serving

## Quick Start

```bash
# Clone and setup
git clone https://github.com/austintechreviews/DiffusionLLMThing.git
cd DiffusionLLMThing
./setup.sh --dev

# Prepare your data
python scripts/prepare_data.py --input data/raw_text/ --output data/processed

# Train a model
python scripts/train.py --data-dir data/processed --model-preset base

# Generate text
python scripts/sample.py --checkpoint checkpoints/checkpoint_final.pt --num-samples 5

# Chat interactively
python scripts/chat.py --checkpoint checkpoints/checkpoint_final.pt

# Start API server
python scripts/server.py --checkpoint checkpoints/checkpoint_final.pt --port 8000
```

## Installation

### Using setup script (recommended)

```bash
./setup.sh --all    # Install all dependencies
source venv/bin/activate
```

### Manual installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 7.0+ GPU (recommended for training)

## Data Preparation

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

### Basic training

```bash
python scripts/train.py \
    --data-dir data/processed \
    --model-preset base \
    --batch-size 32 \
    --max-steps 100000
```

### Model presets

| Preset  | Params | Hidden | Layers | Heads | Max Len |
|---------|--------|--------|--------|-------|---------|
| tiny    | ~5M    | 128    | 2      | 4     | 256     |
| small   | ~20M   | 256    | 4      | 8     | 512     |
| base    | ~85M   | 512    | 6      | 8     | 512     |
| medium  | ~200M  | 768    | 12     | 12    | 1024    |
| large   | ~400M  | 1024   | 16     | 16    | 1024    |
| xl      | ~1.5B  | 2048   | 24     | 16    | 2048    |

### Advanced options

```bash
python scripts/train.py \
    --data-dir data/processed \
    --model-preset base \
    --batch-size 16 \
    --grad-accum 4 \          # Effective batch: 16*4=64
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

### Resuming training

```bash
python scripts/train.py \
    --data-dir data/processed \
    --resume checkpoints/checkpoint_step_005000.pt
```

### Test mode

```bash
python scripts/train.py --test  # Tiny model, 100 steps
```

## Generation

### Sample generation

```bash
python scripts/sample.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --num-samples 10 \
    --max-length 256 \
    --temperature 0.8 \
    --output samples.txt
```

### Interactive chat

```bash
python scripts/chat.py --checkpoint checkpoints/checkpoint_final.pt

# In chat:
/len 256        # Set max length
/temp 0.7       # Set temperature
/progress       # Toggle progress bar
```

### API Server

```bash
# Start server
python scripts/server.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --port 8000

# Query API
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Once upon a time", "max_length": 128, "temperature": 0.8}'
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model` | GET | Model information |
| `/generate` | POST | Generate text |
| `/load` | POST | Load checkpoint |

#### Example API request

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

## Configuration

### YAML config file

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
├── setup.sh
├── run.sh
└── requirements.txt
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

# Load config and create model
config = get_model_config("base")
model = DiscreteDiffusionTransformer(config)

# Load tokenizer
tokenizer = DiffusionTokenizer.load("data/processed/tokenizer.json")

# Encode text
ids = tokenizer.encode("Hello, world!", add_bos=True, add_eos=True)

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

## License

MIT License

## References

- [Mercury: Fast Token Generation with Discrete Diffusion](https://arxiv.org/abs/2401.xxxxx)
- [MDLM: Masked Diffusion Language Models](https://arxiv.org/abs/2402.xxxxx)
- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
