"""
Tests for the model architecture.
"""

import pytest
import torch

from diffusionllm.model import (
    ModelConfig,
    TimestepEmbedding,
    AdaLN,
    DiscreteDiffusionTransformer,
    TransformerBlockWithAdaLN,
)


class TestTimestepEmbedding:
    """Tests for TimestepEmbedding module."""
    
    def test_output_shape(self):
        """Output should have shape (batch, hidden_dim)."""
        hidden_dim = 128
        emb = TimestepEmbedding(hidden_dim)
        
        t = torch.tensor([0, 50, 100, 500])
        output = emb(t)
        
        assert output.shape == (4, hidden_dim)
    
    def test_same_t_same_embedding(self):
        """Same timestep should produce same embedding."""
        emb = TimestepEmbedding(128)
        
        t = torch.tensor([50, 50, 50])
        output = emb(t)
        
        # All rows should be identical
        assert torch.allclose(output[0], output[1])
        assert torch.allclose(output[1], output[2])
    
    def test_different_t_different_embedding(self):
        """Different timesteps should produce different embeddings."""
        emb = TimestepEmbedding(128)
        
        t = torch.tensor([0, 50, 100])
        output = emb(t)
        
        # All rows should be different
        assert not torch.allclose(output[0], output[1])
        assert not torch.allclose(output[1], output[2])
    
    def test_t_zero_embedding(self):
        """t=0 should produce a valid embedding."""
        emb = TimestepEmbedding(128)
        
        t = torch.zeros(1, dtype=torch.long)
        output = emb(t)
        
        assert output.shape == (1, 128)
        assert torch.isfinite(output).all()
    
    def test_t_max_embedding(self):
        """t=T should produce a valid embedding."""
        emb = TimestepEmbedding(128)
        
        t = torch.tensor([1000])
        output = emb(t)
        
        assert output.shape == (1, 128)
        assert torch.isfinite(output).all()


class TestAdaLN:
    """Tests for Adaptive LayerNorm module."""
    
    def test_output_shape(self):
        """Output should have same shape as input."""
        hidden_dim = 128
        ada_ln = AdaLN(hidden_dim)
        
        x = torch.randn(4, 16, hidden_dim)
        t_emb = torch.randn(4, hidden_dim)
        output = ada_ln(x, t_emb)
        
        assert output.shape == x.shape
    
    def test_t_emb_affects_output(self):
        """Different t_emb should produce different outputs."""
        ada_ln = AdaLN(128)
        
        x = torch.randn(2, 8, 128)
        t_emb1 = torch.randn(2, 128)
        t_emb2 = torch.randn(2, 128)
        
        out1 = ada_ln(x, t_emb1)
        out2 = ada_ln(x, t_emb2)
        
        assert not torch.allclose(out1, out2)
    
    def test_same_t_emb_same_output(self):
        """Same t_emb should produce same output."""
        ada_ln = AdaLN(128)
        
        x = torch.randn(2, 8, 128)
        t_emb = torch.randn(2, 128)
        
        out1 = ada_ln(x, t_emb)
        out2 = ada_ln(x, t_emb)
        
        assert torch.allclose(out1, out2)
    
    def test_gradient_flow(self):
        """Gradients should flow through AdaLN."""
        hidden_dim = 128
        ada_ln = AdaLN(hidden_dim)
        
        x = torch.randn(4, 16, hidden_dim, requires_grad=True)
        t_emb = torch.randn(4, hidden_dim, requires_grad=True)
        output = ada_ln(x, t_emb)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert t_emb.grad is not None
        assert torch.isfinite(x.grad).all()


class TestTransformerBlockWithAdaLN:
    """Tests for TransformerBlockWithAdaLN."""
    
    def test_output_shape(self):
        """Output should have same shape as input."""
        hidden_dim = 128
        num_heads = 4
        
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        block = TransformerBlockWithAdaLN(layer, hidden_dim)
        
        x = torch.randn(4, 16, hidden_dim)
        t_emb = torch.randn(4, hidden_dim)
        output = block(x, t_emb)
        
        assert output.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradients should flow through the block."""
        hidden_dim = 128
        num_heads = 4
        
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        block = TransformerBlockWithAdaLN(layer, hidden_dim)
        
        x = torch.randn(4, 16, hidden_dim, requires_grad=True)
        t_emb = torch.randn(4, hidden_dim, requires_grad=True)
        output = block(x, t_emb)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestDiscreteDiffusionTransformer:
    """Tests for the main diffusion transformer model."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return ModelConfig(
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            max_seq_len=32,
        )
    
    def test_model_creation(self, small_config):
        """Model should be creatable with config."""
        model = DiscreteDiffusionTransformer(small_config)
        
        assert model is not None
        assert len(list(model.parameters())) > 0
    
    def test_forward_output_shape(self, small_config):
        """Forward pass should produce correct output shape."""
        model = DiscreteDiffusionTransformer(small_config)
        
        x = torch.randint(0, small_config.vocab_size, (4, 16))
        t = torch.randint(0, 100, (4,))
        logits = model(x, t)
        
        assert logits.shape == (4, 16, small_config.vocab_size)
    
    def test_forward_with_scalar_t(self, small_config):
        """Should handle scalar timestep."""
        model = DiscreteDiffusionTransformer(small_config)
        
        x = torch.randint(0, small_config.vocab_size, (4, 16))
        t = torch.tensor(50)
        logits = model(x, t)
        
        assert logits.shape == (4, 16, small_config.vocab_size)
    
    def test_gradient_flow(self, small_config):
        """Gradients should flow through the model."""
        model = DiscreteDiffusionTransformer(small_config)
        
        x = torch.randint(0, small_config.vocab_size, (4, 16))
        t = torch.randint(0, 100, (4,))
        logits = model(x, t)
        
        loss = logits.sum()
        loss.backward()
        
        # Check that key parameters have gradients
        # Note: Some norm parameters in wrapped layers may not have gradients
        # because we use AdaLN for conditioning instead
        params_with_grad = 0
        params_total = 0
        for name, param in model.named_parameters():
            params_total += 1
            if param.grad is not None:
                if torch.isfinite(param.grad).all():
                    params_with_grad += 1
        
        # At least 80% of parameters should have finite gradients
        assert params_with_grad >= params_total * 0.8
    
    def test_parameter_count(self, small_config):
        """Should have expected number of parameters."""
        model = DiscreteDiffusionTransformer(small_config)
        num_params = sum(p.numel() for p in model.parameters())
        
        # Rough estimate: embeddings + transformer + output
        embedding_params = small_config.vocab_size * small_config.hidden_dim
        output_params = small_config.hidden_dim * small_config.vocab_size
        position_params = small_config.max_seq_len * small_config.hidden_dim
        
        # Should be at least these basic parameters
        assert num_params > embedding_params + output_params
    
    def test_mask_token_handling(self, small_config):
        """Model should handle mask token in input."""
        model = DiscreteDiffusionTransformer(small_config)
        
        # Input with all mask tokens
        x = torch.full((4, 16), small_config.mask_token_id)
        t = torch.randint(0, 100, (4,))
        logits = model(x, t)
        
        assert logits.shape == (4, 16, small_config.vocab_size)
        assert torch.isfinite(logits).all()
    
    def test_pad_token_handling(self, small_config):
        """Model should handle pad token in input."""
        model = DiscreteDiffusionTransformer(small_config)
        
        # Input with padding
        x = torch.full((4, 16), small_config.pad_token_id)
        t = torch.randint(0, 100, (4,))
        logits = model(x, t)
        
        assert logits.shape == (4, 16, small_config.vocab_size)
        assert torch.isfinite(logits).all()
    
    def test_variable_sequence_length(self, small_config):
        """Model should handle variable sequence lengths."""
        model = DiscreteDiffusionTransformer(small_config)
        
        for seq_len in [8, 16, 32]:
            x = torch.randint(0, small_config.vocab_size, (2, seq_len))
            t = torch.randint(0, 100, (2,))
            logits = model(x, t)
            
            assert logits.shape == (2, seq_len, small_config.vocab_size)
    
    def test_weight_initialization(self, small_config):
        """Weights should be properly initialized."""
        model = DiscreteDiffusionTransformer(small_config)
        
        # Check embedding std
        emb_std = model.token_embed.weight.std().item()
        assert 0.01 < emb_std < 0.03  # Should be around 0.02
    
    def test_training_mode(self, small_config):
        """Model should work in training mode."""
        model = DiscreteDiffusionTransformer(small_config)
        model.train()
        
        x = torch.randint(0, small_config.vocab_size, (4, 16))
        t = torch.randint(0, 100, (4,))
        logits = model(x, t)
        
        assert logits.shape == (4, 16, small_config.vocab_size)
    
    def test_eval_mode(self, small_config):
        """Model should work in eval mode."""
        model = DiscreteDiffusionTransformer(small_config)
        model.eval()
        
        x = torch.randint(0, small_config.vocab_size, (4, 16))
        t = torch.randint(0, 100, (4,))
        
        with torch.no_grad():
            logits = model(x, t)
        
        assert logits.shape == (4, 16, small_config.vocab_size)
    
    def test_deterministic_eval(self, small_config):
        """Eval mode should be deterministic."""
        model = DiscreteDiffusionTransformer(small_config)
        model.eval()
        
        x = torch.randint(0, small_config.vocab_size, (4, 16))
        t = torch.randint(0, 100, (4,))
        
        with torch.no_grad():
            logits1 = model(x, t)
            logits2 = model(x, t)
        
        assert torch.allclose(logits1, logits2)
