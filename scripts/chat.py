#!/usr/bin/env python3
"""
Interactive chat script for discrete diffusion language model.

Usage:
    python scripts/chat.py --checkpoint checkpoints/checkpoint_final.pt
    python scripts/chat.py --checkpoint checkpoints/checkpoint_final.pt --max-length 256
"""

import argparse
import os
import sys
import readline  # For better input handling

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
    )
    
    model = DiscreteDiffusionTransformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    T = config.get('T', 1000)
    
    return model, model_config, T


def generate_response(
    model: torch.nn.Module,
    model_config: ModelConfig,
    T: int,
    prompt: str,
    max_length: int,
    temperature: float,
    device: torch.device,
    show_progress: bool = False,
) -> str:
    """Generate a response given a prompt."""
    # Convert prompt to tokens (simple encoding for now)
    # In production, use the same tokenizer as training
    prompt_tokens = [min(ord(c) % (model_config.vocab_size - 3) + 3, model_config.vocab_size - 1) 
                     for c in prompt[:max_length]]
    
    # Pad to max_length if needed
    if len(prompt_tokens) < max_length:
        prompt_tokens.extend([model_config.pad_token_id] * (max_length - len(prompt_tokens)))
    
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Generate using diffusion
    progress_callback = None
    if show_progress:
        pbar = None
        def callback(step, total):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc="Generating", leave=False)
            pbar.update(1)
        progress_callback = callback
    
    with torch.no_grad():
        generated = sample(
            model=model,
            T=T,
            mask_token_id=model_config.mask_token_id,
            batch_size=1,
            seq_len=max_length,
            temperature=temperature,
            unmask_schedule="linear",
            device=device,
            progress_callback=progress_callback,
        )
    
    if show_progress and pbar is not None:
        pbar.close()
    
    # Convert to text
    texts = tokens_to_text(generated, eos_token_id=model_config.eos_token_id)
    return texts[0]


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  Discrete Diffusion Language Model - Interactive Chat")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit, /exit, /q  - Exit the chat")
    print("  /help, /h         - Show this help message")
    print("  /temp <value>     - Set temperature (0.1-2.0, higher = more creative)")
    print("  /len <value>      - Set max response length (32-512)")
    print("  /progress         - Toggle generation progress bar")
    print("\n" + "-" * 60)


def print_help():
    """Print help message."""
    print("\nAvailable commands:")
    print("  /quit, /exit, /q  - Exit the chat")
    print("  /help, /h         - Show this help message")
    print("  /temp <value>     - Set temperature (0.1-2.0)")
    print("  /len <value>      - Set max response length (32-512)")
    print("  /progress         - Toggle generation progress bar")
    print("  /model            - Show model info")
    print("  /clear            - Clear chat history")
    print()


def main(args):
    """Main interactive chat function."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # Load model
    model, model_config, T = load_model(args.checkpoint, device)
    
    # Adjust T if specified
    if args.T is not None:
        T = args.T
    
    print(f"Model loaded successfully!")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Num layers: {model_config.num_layers}")
    print(f"  Diffusion steps: {T}")
    
    # Chat state
    temperature = args.temperature
    max_length = args.max_length
    show_progress = args.progress
    chat_history = []
    
    # Print banner
    print_banner()
    
    print("\nModel: Hello! I'm a diffusion-based language model. How can I help you today?")
    print("You: ", end="", flush=True)
    
    while True:
        try:
            # Get user input
            user_input = input().strip()
            
            if not user_input:
                print("You: ", end="", flush=True)
                continue
            
            # Check for commands
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                
                if cmd in ['/quit', '/exit', '/q']:
                    print("\nGoodbye!")
                    break
                
                elif cmd in ['/help', '/h']:
                    print_help()
                
                elif cmd == '/temp' and len(parts) > 1:
                    try:
                        new_temp = float(parts[1])
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"Temperature set to {temperature}")
                        else:
                            print("Temperature must be between 0.1 and 2.0")
                    except ValueError:
                        print("Invalid temperature value")
                
                elif cmd == '/len' and len(parts) > 1:
                    try:
                        new_len = int(parts[1])
                        if 32 <= new_len <= 512:
                            max_length = new_len
                            print(f"Max length set to {max_length}")
                        else:
                            print("Max length must be between 32 and 512")
                    except ValueError:
                        print("Invalid length value")
                
                elif cmd == '/progress':
                    show_progress = not show_progress
                    print(f"Progress bar: {'ON' if show_progress else 'OFF'}")
                
                elif cmd == '/model':
                    print(f"\nModel Configuration:")
                    print(f"  Vocab size: {model_config.vocab_size}")
                    print(f"  Hidden dim: {model_config.hidden_dim}")
                    print(f"  Num layers: {model_config.num_layers}")
                    print(f"  Num heads: {model_config.num_heads}")
                    print(f"  Diffusion steps: {T}")
                    print(f"  Temperature: {temperature}")
                    print(f"  Max length: {max_length}")
                    print()
                
                elif cmd == '/clear':
                    chat_history = []
                    print("Chat history cleared.\n")
                
                else:
                    print(f"Unknown command: {cmd}. Type /help for commands.")
                
                print("You: ", end="", flush=True)
                continue
            
            # Add to chat history
            chat_history.append(f"User: {user_input}")
            
            # Generate response
            print("\nModel: ", end="", flush=True)
            
            # Create prompt from chat history (last few turns)
            context = "\n".join(chat_history[-4:])  # Last 2 exchanges
            prompt = context + "\nModel: "
            
            response = generate_response(
                model=model,
                model_config=model_config,
                T=T,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                device=device,
                show_progress=show_progress,
            )
            
            print(response)
            
            # Add response to history
            chat_history.append(f"Model: {response}")
            print("\nYou: ", end="", flush=True)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit or continue typing.")
            print("You: ", end="", flush=True)
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("You: ", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive chat with Discrete Diffusion LM")
    
    # Required
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    
    # Generation parameters
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
        "--progress", action="store_true",
        help="Show generation progress bar"
    )
    
    args = parser.parse_args()
    main(args)
