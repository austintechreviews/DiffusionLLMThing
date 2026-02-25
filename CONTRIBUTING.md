# Contributing to DiffusionLLM

Thank you for considering contributing to DiffusionLLM! This document provides guidelines and instructions for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/austintechreviews/DiffusionLLMThing.git
cd DiffusionLLMThing

# Run interactive setup with dev dependencies
./setup.sh -i  # Select "Development" installation type

# Or manually
./setup.sh --dev
source venv/bin/activate
```

## Running Tests

```bash
# Run all tests
./run.sh test

# Or directly
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=diffusionllm --cov-report=html

# Run specific test file
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::TestDiscreteDiffusionTransformer::test_forward_output_shape -v
```

## Code Style

This project uses Black for formatting and Ruff for linting:

```bash
# Format code
black diffusionllm/ scripts/

# Lint code
ruff check diffusionllm/ scripts/

# Auto-fix lint issues
ruff check --fix diffusionllm/ scripts/
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make changes** and ensure tests pass:
   ```bash
   ./run.sh test
   ```
4. **Format and lint** your code:
   ```bash
   black diffusionllm/ scripts/
   ruff check --fix diffusionllm/ scripts/
   ```
5. **Commit** with clear messages:
   ```bash
   git commit -m "Add feature: description of feature"
   ```
6. **Push** and create a Pull Request

## Code Organization

```
diffusionllm/
├── config.py      # Model configurations and presets
├── model.py       # Neural network architectures
├── diffusion.py   # Diffusion process functions
├── sampling.py    # Generation and inference
├── tokenizer.py   # Tokenization utilities
├── data.py        # Dataset classes
└── utils.py       # Training utilities
```

## Adding Features

### New Model Architecture

1. Create class in `diffusionllm/model.py`
2. Add config options to `diffusionllm/config.py`
3. Add tests in `tests/test_model.py`

### New Sampling Strategy

1. Add function to `diffusionllm/sampling.py`
2. Export in `diffusionllm/__init__.py`
3. Add tests in `tests/test_sampling.py`

### New Dataset Format

1. Create Dataset class in `diffusionllm/data.py`
2. Update `scripts/prepare_data.py` if needed
3. Add tests in `tests/test_data.py`

## Documentation

- Update README.md for user-facing changes
- Add docstrings to public functions and classes
- Include type hints for function arguments and return values

Example docstring:
```python
def sample(
    model: torch.nn.Module,
    T: int,
    mask_token_id: int,
    batch_size: int = 1,
    seq_len: int = 128,
) -> torch.Tensor:
    """
    Generate text by iterative denoising.
    
    Args:
        model: Trained diffusion model
        T: Number of diffusion steps
        mask_token_id: Token ID for [MASK]
        batch_size: Number of sequences to generate
        seq_len: Length of each sequence
    
    Returns:
        Generated token ids of shape (batch_size, seq_len)
    """
```

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

## Feature Requests

Feature requests are welcome! Please include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Potential impact on existing features

## Questions?

Open an issue for any questions or discussions about the project.
