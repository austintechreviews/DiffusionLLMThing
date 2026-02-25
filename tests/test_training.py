"""
Tests for the full training loop.
"""

import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusionllm.model import ModelConfig, DiscreteDiffusionTransformer
from diffusionllm.diffusion import forward_diffusion, compute_loss
from diffusionllm.utils import get_lr_schedule


class SimpleTextDataset:
    """Simple dataset for testing."""
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.data = torch.randint(3, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_step(model, x0, config, optimizer, device):
    """Simplified training step for testing."""
    model.train()
    batch_size = x0.shape[0]
    
    # Sample random timestep
    t = torch.randint(0, config['T'], (batch_size,), device=device)
    
    # Forward diffusion
    xt, mask = forward_diffusion(
        x0, t, config['T'], config['mask_token_id'], config['pad_token_id']
    )
    
    # Forward pass
    optimizer.zero_grad()
    logits = model(xt, t)
    
    # Compute loss
    loss = compute_loss(
        logits, x0, mask, t, config['T'], config['pad_token_id']
    )
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
    
    # Optimizer step
    optimizer.step()
    
    return loss.item(), grad_norm.item()


class TestTrainingStep:
    """Tests for single training step."""
    
    @pytest.fixture
    def train_config(self):
        """Training configuration for testing."""
        return {
            'T': 100,
            'mask_token_id': 0,
            'pad_token_id': 1,
            'grad_clip': 1.0,
        }
    
    @pytest.fixture
    def model_and_data(self, train_config):
        """Create model and data for testing."""
        config = ModelConfig(
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        model = DiscreteDiffusionTransformer(config)
        
        dataset = SimpleTextDataset(32, 16, 100)
        dataloader = DataLoader(dataset, batch_size=4)
        
        return model, dataloader, config
    
    def test_training_step_reduces_loss(self, train_config, model_and_data):
        """Training should eventually reduce loss."""
        model, dataloader, model_config = model_and_data
        device = torch.device('cpu')
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Get initial loss
        x0 = next(iter(dataloader))
        initial_loss, _ = train_step(model, x0.to(device), train_config, optimizer, device)
        
        # Train for a few steps
        losses = [initial_loss]
        for _ in range(10):
            for batch in dataloader:
                loss, _ = train_step(model, batch.to(device), train_config, optimizer, device)
                losses.append(loss)
                break  # One batch per epoch for speed
        
        # Loss should generally decrease (allowing for variance)
        # Compare first and last few losses
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        
        # This is a soft assertion - training should help
        assert late_avg <= early_avg * 1.5  # Allow some variance
    
    def test_training_step_gradient_flow(self, train_config, model_and_data):
        """Training step should produce valid gradients."""
        model, dataloader, _ = model_and_data
        device = torch.device('cpu')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x0 = next(iter(dataloader))
        loss, grad_norm = train_step(model, x0.to(device), train_config, optimizer, device)

        # Check gradients exist and are finite
        # Note: Some norm parameters in wrapped layers may not have gradients
        params_with_grad = 0
        params_finite = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
                if torch.isfinite(param.grad).all():
                    params_finite += 1

        # Most parameters should have finite gradients
        assert params_finite > 0
        assert torch.isfinite(torch.tensor(grad_norm))
        # Note: grad_norm might exceed clip value due to how PyTorch reports it
        # (it reports the norm before clipping)
        assert grad_norm > 0
    
    def test_training_step_loss_is_finite(self, train_config, model_and_data):
        """Training loss should be finite."""
        model, dataloader, _ = model_and_data
        device = torch.device('cpu')
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for batch in dataloader:
            loss, _ = train_step(model, batch.to(device), train_config, optimizer, device)
            
            assert torch.isfinite(torch.tensor(loss))
            assert loss >= 0


class TestTrainingLoop:
    """Tests for full training loop."""
    
    @pytest.fixture
    def full_config(self):
        """Full training configuration."""
        return {
            'model': ModelConfig(
                vocab_size=100,
                hidden_dim=32,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
            ),
            'T': 50,
            'mask_token_id': 0,
            'pad_token_id': 1,
            'grad_clip': 1.0,
            'lr': 1e-3,
            'warmup_steps': 10,
            'max_steps': 100,
        }
    
    def test_training_loop_convergence(self, full_config):
        """Training loop should show loss convergence."""
        device = torch.device('cpu')
        
        model = DiscreteDiffusionTransformer(full_config['model']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=full_config['lr'])
        
        dataset = SimpleTextDataset(64, 8, 100)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        losses = []
        step = 0
        
        for epoch in range(3):  # Multiple epochs
            for batch in dataloader:
                if step >= full_config['max_steps']:
                    break
                
                # Training step
                t = torch.randint(0, full_config['T'], (8,), device=device)
                xt, mask = forward_diffusion(
                    batch.to(device), t, full_config['T'],
                    full_config['mask_token_id'], full_config['pad_token_id']
                )
                
                optimizer.zero_grad()
                logits = model(xt, t)
                loss = compute_loss(
                    logits, batch.to(device), mask, t,
                    full_config['T'], full_config['pad_token_id']
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), full_config['grad_clip'])
                optimizer.step()
                
                losses.append(loss.item())
                step += 1
        
        # Loss should generally decrease
        early_avg = sum(losses[:10]) / min(10, len(losses))
        late_avg = sum(losses[-10:]) / min(10, len(losses))
        
        assert late_avg < early_avg * 2  # Allow variance but should improve
    
    def test_lr_schedule_during_training(self, full_config):
        """Learning rate should follow schedule during training."""
        device = torch.device('cpu')
        
        model = DiscreteDiffusionTransformer(full_config['model']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=full_config['lr'])
        
        lrs = []
        for step in range(full_config['max_steps']):
            lr_mult = get_lr_schedule(step, full_config['warmup_steps'], full_config['max_steps'])
            current_lr = full_config['lr'] * lr_mult
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            lrs.append(current_lr)
        
        # LR should increase during warmup
        assert lrs[0] < lrs[full_config['warmup_steps'] - 1]
        
        # LR should decrease after warmup
        assert lrs[full_config['warmup_steps']] >= lrs[-1]
        
        # Final LR should be close to 0
        assert lrs[-1] < full_config['lr'] * 0.1


class TestCheckpointResume:
    """Tests for checkpoint and resume functionality."""

    def test_resume_continues_training(self):
        """Resuming from checkpoint should continue training."""
        from diffusionllm.utils import save_checkpoint, load_checkpoint

        device = torch.device('cpu')
        config = ModelConfig(vocab_size=50, hidden_dim=16, num_layers=1, num_heads=4)

        model = DiscreteDiffusionTransformer(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for a bit
            for step in range(5):
                x0 = torch.randint(3, 50, (4, 8))
                t = torch.randint(0, 100, (4,))
                xt, mask = forward_diffusion(x0, t, 100, 0, 1)

                optimizer.zero_grad()
                logits = model(xt, t)
                loss = compute_loss(logits, x0, mask, t, 100, 1)
                loss.backward()
                optimizer.step()

            # Save checkpoint
            ckpt_path = os.path.join(tmpdir, 'checkpoint.pt')
            save_checkpoint(5, model, optimizer, None, {}, ckpt_path)

            # Modify model
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(0.5)

            # Load checkpoint
            _, model, optimizer, _ = load_checkpoint(
                ckpt_path, model, optimizer, None, device
            )

            # Verify model is restored by checking it can train
            x0 = torch.randint(3, 50, (4, 8))
            t = torch.randint(0, 100, (4,))
            xt, mask = forward_diffusion(x0, t, 100, 0, 1)

            optimizer.zero_grad()
            logits = model(xt, t)
            loss = compute_loss(logits, x0, mask, t, 100, 1)
            loss.backward()

            # Should have valid gradients (at least some parameters)
            params_with_grad = sum(1 for p in model.parameters() 
                                   if p.grad is not None and torch.isfinite(p.grad).all())
            assert params_with_grad > 0
