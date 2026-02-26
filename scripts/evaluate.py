#!/usr/bin/env python3
"""
Evaluation script for discrete diffusion language model.

Computes perplexity and other metrics on a test dataset.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_final.pt --data-path data/test.txt
"""

import argparse
import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusionllm.model import ModelConfig, DiscreteDiffusionTransformer
from diffusionllm.diffusion import forward_diffusion, compute_loss, get_noise_schedule


class TextDataset:
    """Simple text dataset for evaluation."""
    
    def __init__(self, data_path: str, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> torch.Tensor:
        """Load and tokenize text data."""
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                lines = f.readlines()
            
            all_tokens = []
            for line in lines:
                tokens = [min(ord(c) % (self.vocab_size - 3) + 3, self.vocab_size - 1) 
                         for c in line.strip()]
                all_tokens.extend(tokens)
                all_tokens.append(2)  # EOS
            
            data = []
            for i in range(0, len(all_tokens) - self.seq_len, self.seq_len):
                data.append(all_tokens[i:i + self.seq_len])
            
            return torch.tensor(data, dtype=torch.long)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    model_config: ModelConfig,
    T: int,
    device: torch.device,
    alpha: torch.Tensor,
) -> dict:
    """
    Evaluate model on dataset.
    
    Returns:
        Dictionary with metrics (loss, perplexity, etc.)
    """
    model.eval()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    num_batches = 0
    num_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x0 = batch.to(device)
            batch_size = x0.shape[0]
            
            # Sample timesteps
            t = torch.randint(0, T, (batch_size,), device=device)
            
            # Forward diffusion
            xt, mask = forward_diffusion(
                x0, t, T,
                model_config.mask_token_id,
                model_config.pad_token_id,
                alpha=alpha,
            )
            
            # Forward pass
            logits = model(xt, t)
            
            # Compute loss
            loss = compute_loss(
                logits, x0, mask, t, T,
                model_config.pad_token_id,
                alpha=alpha,
            )
            
            # Compute raw cross-entropy (for perplexity)
            if mask.sum() > 0:
                ce_loss = torch.nn.functional.cross_entropy(
                    logits[mask], x0[mask]
                )
                total_ce_loss += ce_loss.item() * mask.sum().item()
                num_tokens += mask.sum().item()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_tokens if num_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()
    
    return {
        'loss': avg_loss,
        'cross_entropy': avg_ce_loss,
        'perplexity': perplexity,
        'num_tokens': num_tokens,
    }


def evaluate_at_timesteps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    model_config: ModelConfig,
    T: int,
    device: torch.device,
    timesteps: List[int],
) -> dict:
    """
    Evaluate model at specific timesteps.
    
    Shows how performance varies with noise level.
    """
    model.eval()
    alpha = get_noise_schedule(T)
    
    results = {}
    
    with torch.no_grad():
        for t_val in timesteps:
            total_loss = 0.0
            total_ce = 0.0
            num_tokens = 0
            num_batches = 0
            
            for batch in dataloader:
                x0 = batch.to(device)
                batch_size = x0.shape[0]
                
                t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
                
                xt, mask = forward_diffusion(
                    x0, t, T,
                    model_config.mask_token_id,
                    model_config.pad_token_id,
                )
                
                logits = model(xt, t)
                
                if mask.sum() > 0:
                    ce = torch.nn.functional.cross_entropy(logits[mask], x0[mask])
                    total_ce += ce.item() * mask.sum().item()
                    num_tokens += mask.sum().item()
                
                num_batches += 1
            
            avg_ce = total_ce / num_tokens if num_tokens > 0 else float('inf')
            results[t_val] = {
                'cross_entropy': avg_ce,
                'perplexity': torch.exp(torch.tensor(avg_ce)).item() if num_tokens > 0 else float('inf'),
            }
    
    return results


def main(args):
    """Main evaluation function."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

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

    T = config.get('T', 1000)

    # Initialize model
    model = DiscreteDiffusionTransformer(model_config).to(device)
    # Load with strict=False to handle rotary embedding buffers
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"Model loaded (step {checkpoint.get('step', 'unknown')})")
    
    # Setup data
    print(f"Loading test data: {args.data_path}")
    dataset = TextDataset(
        data_path=args.data_path,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    print(f"Test set size: {len(dataset)} sequences")
    
    # Get noise schedule
    alpha = get_noise_schedule(T)
    
    # Full evaluation
    print("\n" + "=" * 60)
    print("FULL EVALUATION")
    print("=" * 60)
    
    metrics = evaluate(model, dataloader, model_config, T, device, alpha)
    
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Cross-Entropy: {metrics['cross_entropy']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Tokens evaluated: {metrics['num_tokens']:,}")
    
    # Evaluation at different timesteps
    if args.eval_timesteps:
        print("\n" + "=" * 60)
        print("EVALUATION BY TIMESTEP")
        print("=" * 60)
        
        timesteps = [0, 100, 200, 400, 600, 800, 999]
        results = evaluate_at_timesteps(model, dataloader, model_config, T, device, timesteps)
        
        print(f"{'Timestep':>10} | {'Cross-Entropy':>15} | {'Perplexity':>12}")
        print("-" * 42)
        for t_val in timesteps:
            r = results[t_val]
            print(f"{t_val:>10} | {r['cross_entropy']:>15.4f} | {r['perplexity']:>12.2f}")
    
    # Save results
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        
        results = {
            'checkpoint': args.checkpoint,
            'data_path': args.data_path,
            'metrics': metrics,
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Discrete Diffusion Language Model")
    
    # Required
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to test data"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length for evaluation"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--eval-timesteps", action="store_true",
        help="Also evaluate at different timesteps"
    )
    
    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save evaluation results (JSON)"
    )
    
    args = parser.parse_args()
    main(args)
