#!/usr/bin/env python3
"""
FastAPI server for discrete diffusion language model.

Provides REST API for text generation.

Usage:
    python scripts/server.py --checkpoint checkpoints/checkpoint_final.pt --port 8000
    python scripts/server.py --checkpoint checkpoints/checkpoint_final.pt --port 8000 --host 0.0.0.0
"""

import argparse
import os
import sys
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusionllm.config import get_model_config
from diffusionllm.model import DiscreteDiffusionTransformer
from diffusionllm.tokenizer import DiffusionTokenizer
from diffusionllm.sampling import sample, tokens_to_text


# =============================================================================
# API Models
# =============================================================================

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt text")
    max_length: int = Field(default=128, ge=16, le=2048, description="Maximum generation length")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.1, le=1.0, description="Nucleus sampling threshold")
    num_sequences: int = Field(default=1, ge=1, le=10, description="Number of sequences to generate")
    diffusion_steps: Optional[int] = Field(default=None, ge=10, le=1000, description="Diffusion steps (overrides checkpoint)")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    prompt: str
    generations: List[str]
    model_info: dict


class ModelInfo(BaseModel):
    """Model information."""
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    diffusion_steps: int
    device: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


# =============================================================================
# Server
# =============================================================================

app = FastAPI(
    title="Discrete Diffusion LM API",
    description="REST API for discrete diffusion language model",
    version="0.2.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.model_config = None
        self.tokenizer = None
        self.T = 1000
        self.device = torch.device('cpu')
        self.loaded = False


state = ModelState()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model and tokenizer from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Load or create tokenizer
    tokenizer_path = config.get('tokenizer_path', '')
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = DiffusionTokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer: {tokenizer.actual_vocab_size} tokens")
    else:
        tokenizer = None
        print("No tokenizer found, using character-level encoding")
    
    # Create model config
    model_config = get_model_config(
        preset=config.get('model_preset', 'base'),
        vocab_size=config.get('vocab_size', 32000),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        use_rotary_embeddings=config.get('use_rotary_embeddings', False),  # Read from checkpoint
    )

    # Create and load model
    model = DiscreteDiffusionTransformer(model_config).to(device)
    # Load with strict=False to handle rotary embedding buffers
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    T = config.get('T', 1000)
    
    print(f"Model loaded: {model_config.num_parameters_millions:.2f}M parameters")
    
    return model, model_config, tokenizer, T


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    if state.loaded:
        return
    
    # Setup device
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1).cuda()
            (x + x).item()
            state.device = torch.device('cuda')
        except Exception:
            state.device = torch.device('cpu')
    else:
        state.device = torch.device('cpu')
    
    print(f"Using device: {state.device}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.loaded,
        device=str(state.device),
    )


@app.get("/model", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        vocab_size=state.model_config.vocab_size,
        hidden_dim=state.model_config.hidden_dim,
        num_layers=state.model_config.num_layers,
        num_heads=state.model_config.num_heads,
        max_seq_len=state.model_config.max_seq_len,
        diffusion_steps=state.T,
        device=str(state.device),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text given a prompt."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode prompt
        if state.tokenizer:
            prompt_ids = state.tokenizer.encode(
                request.prompt,
                add_bos=True,
                add_eos=False,
                truncation=True,
                max_length=request.max_length,
            )
        else:
            # Character-level fallback
            prompt_ids = [
                min(ord(c) % (state.model_config.vocab_size - 3) + 3, state.model_config.vocab_size - 1)
                for c in request.prompt[:request.max_length]
            ]
        
        # Pad to max_length
        if len(prompt_ids) < request.max_length:
            pad_len = request.max_length - len(prompt_ids)
            prompt_ids.extend([state.model_config.pad_token_id] * pad_len)
        
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=state.device)
        
        # Generate
        T = request.diffusion_steps or state.T
        
        with torch.no_grad():
            generated = sample(
                model=state.model,
                T=T,
                mask_token_id=state.model_config.mask_token_id,
                batch_size=request.num_sequences,
                seq_len=request.max_length,
                temperature=request.temperature,
                unmask_schedule="linear",
                device=state.device,
            )
        
        # Decode
        if state.tokenizer:
            generations = state.tokenizer.decode_batch(
                generated.tolist(),
                skip_special_tokens=True,
            )
        else:
            generations = tokens_to_text(generated, eos_token_id=state.model_config.eos_token_id)
        
        return GenerateResponse(
            prompt=request.prompt,
            generations=generations,
            model_info={
                "vocab_size": state.model_config.vocab_size,
                "diffusion_steps": T,
                "device": str(state.device),
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load_checkpoint(checkpoint_path: str = Field(..., description="Path to checkpoint")):
    """Load a model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    
    try:
        state.model, state.model_config, state.tokenizer, state.T = load_model(
            checkpoint_path, state.device
        )
        state.loaded = True
        
        return {"status": "success", "message": f"Model loaded from {checkpoint_path}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="FastAPI server for Diffusion LM")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes"
    )
    
    args = parser.parse_args()
    
    # Load model before starting server
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state.model, state.model_config, state.tokenizer, state.T = load_model(args.checkpoint, device)
    state.device = device
    state.loaded = True
    
    # Start server
    import uvicorn
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  GET  /model      - Model information")
    print("  POST /generate   - Generate text")
    print("  POST /load       - Load checkpoint")
    print()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
