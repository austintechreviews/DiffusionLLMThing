#!/usr/bin/env python3
"""
Sampling script for discrete diffusion language model.

Usage:
    python scripts/sample.py --checkpoint checkpoints/checkpoint_final.pt
    python scripts/sample.py --checkpoint checkpoints/checkpoint_final.pt --num-samples 10 --max-length 256
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusionllm.model import ModelConfig, DiscreteDiffusionTransformer
from diffusionllm.sampling import sample, tokens_to_text


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    config = checkpoint.get('config', {})

    model_config = ModelConfig(
        vocab_size=config.get('vocab_size', 32000),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 512),
        mask_token_id=config.get('mask_token_id', 0),
        pad_token_id=config.get('pad_token_id', 1),
        eos_token_id=config.get('eos_token_id', 2),
        use_rotary_embeddings=config.get('use_rotary_embeddings', False),  # Read from checkpoint
    )

    model = DiscreteDiffusionTransformer(model_config).to(device)
    # Load with strict=False to handle rotary embedding buffers
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    T = config.get('T', 1000)

    print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Num layers: {model_config.num_layers}")
    print(f"  Diffusion steps: {T}")

    return model, model_config, T


def main(args):
    """Main sampling function."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, model_config, T = load_model(args.checkpoint, device)
    
    # Adjust T for faster sampling if requested
    if args.T is not None:
        T = args.T
        print(f"Using {T} diffusion steps (overrides checkpoint)")
    
    print(f"\nGenerating {args.num_samples} samples...")
    
    # Progress bar for sampling
    def progress_callback(step, total):
        pbar.update(1)
    
    # Generate samples
    all_texts = []
    
    # Sample in batches
    for batch_start in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - batch_start)
        
        with tqdm(total=T, desc=f"Sampling {batch_start+1}-{batch_start+batch_size}", leave=False) as pbar:
            output = sample(
                model=model,
                T=T,
                mask_token_id=model_config.mask_token_id,
                batch_size=batch_size,
                seq_len=args.max_length,
                temperature=args.temperature,
                unmask_schedule=args.unmask_schedule,
                device=device,
                progress_callback=progress_callback,
            )
        
        # Convert to text
        texts = tokens_to_text(output, eos_token_id=model_config.eos_token_id)
        all_texts.extend(texts)
    
    # Print or save results
    print("\n" + "=" * 60)
    print("GENERATED SAMPLES")
    print("=" * 60)
    
    for i, text in enumerate(all_texts):
        print(f"\n--- Sample {i + 1} ---")
        print(text[:args.truncate_length] if args.truncate_length else text)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            for i, text in enumerate(all_texts):
                f.write(f"--- Sample {i + 1} ---\n")
                f.write(text + "\n\n")
        print(f"\nSaved samples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from Discrete Diffusion Language Model")
    
    # Required
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for sampling"
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (higher = more diverse)"
    )
    parser.add_argument(
        "--T", type=int, default=None,
        help="Number of diffusion steps (overrides checkpoint)"
    )
    parser.add_argument(
        "--unmask-schedule", type=str, default="linear",
        choices=["linear", "cosine", "uniform"],
        help="Unmasking schedule"
    )
    
    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save generated samples"
    )
    parser.add_argument(
        "--truncate-length", type=int, default=200,
        help="Truncate output for display (0 = no truncation)"
    )
    
    args = parser.parse_args()
    main(args)
