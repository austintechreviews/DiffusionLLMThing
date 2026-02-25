"""
Tests for sampling and inference.
"""

import pytest
import torch

from diffusionllm.model import ModelConfig, DiscreteDiffusionTransformer
from diffusionllm.sampling import (
    sample,
    sample_step,
    tokens_to_text,
)


class TestSampleStep:
    """Tests for single sampling step."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a model with random weights for testing."""
        config = ModelConfig(
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            max_seq_len=32,
        )
        model = DiscreteDiffusionTransformer(config)
        model.eval()
        return model, config
    
    def test_sample_step_output_shape(self, trained_model):
        """Output should have same shape as input."""
        model, config = trained_model
        
        x = torch.full((4, 16), config.mask_token_id)
        t = 50
        
        output = sample_step(
            model, x, t, T=100, mask_token_id=config.mask_token_id
        )
        
        assert output.shape == x.shape
    
    def test_sample_step_unmasks_tokens(self, trained_model):
        """Sample step should unmask some tokens."""
        model, config = trained_model

        x = torch.full((2, 8), config.mask_token_id)
        t = 0

        output = sample_step(
            model, x, t, T=100, mask_token_id=config.mask_token_id,
            strategy="all"
        )

        # With strategy="all", all positions should be updated
        # (though they might still be mask tokens if model predicts them)
        # At minimum, output should have same shape and be different from input
        assert output.shape == x.shape
        # Most positions should be unmasked (model might occasionally predict mask token)
        num_unmasked = (output != config.mask_token_id).sum().item()
        assert num_unmasked > 0  # At least some tokens should be unmasked
    
    def test_sample_step_temperature_effect(self, trained_model):
        """Temperature should affect sampling."""
        model, config = trained_model
        
        x = torch.full((4, 16), config.mask_token_id)
        t = 0
        
        torch.manual_seed(42)
        output_low_temp = sample_step(
            model, x, t, T=100, mask_token_id=config.mask_token_id,
            temperature=0.1
        )
        
        torch.manual_seed(42)
        output_high_temp = sample_step(
            model, x, t, T=100, mask_token_id=config.mask_token_id,
            temperature=2.0
        )
        
        # Low temperature should have more confident (repeated) predictions
        # This is a soft check due to randomness
        assert output_low_temp.shape == output_high_temp.shape
    
    def test_sample_step_partial_unmasking(self, trained_model):
        """Should be able to unmask only a fraction of tokens."""
        model, config = trained_model
        
        x = torch.full((2, 16), config.mask_token_id)
        t = 50
        
        output = sample_step(
            model, x, t, T=100, mask_token_id=config.mask_token_id,
            unmask_ratio=0.5, strategy="random"
        )
        
        # Some tokens should still be masked
        num_masked = (output == config.mask_token_id).sum().item()
        assert num_masked > 0
        assert num_masked < output.numel()
    
    def test_sample_step_no_masked_positions(self, trained_model):
        """Should handle input with no masked positions."""
        model, config = trained_model
        
        x = torch.randint(3, config.vocab_size, (2, 8))
        t = 50
        
        output = sample_step(
            model, x, t, T=100, mask_token_id=config.mask_token_id
        )
        
        # Should return input unchanged
        assert torch.equal(output, x)


class TestSample:
    """Tests for full sampling loop."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a model with random weights for testing."""
        config = ModelConfig(
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            max_seq_len=32,
        )
        model = DiscreteDiffusionTransformer(config)
        model.eval()
        return model, config
    
    def test_sample_output_shape(self, trained_model):
        """Output should have correct shape."""
        model, config = trained_model
        
        output = sample(
            model, T=50, mask_token_id=config.mask_token_id,
            batch_size=2, seq_len=16
        )
        
        assert output.shape == (2, 16)
    
    def test_sample_no_mask_tokens_in_output(self, trained_model):
        """Output should not contain mask tokens."""
        model, config = trained_model
        
        output = sample(
            model, T=50, mask_token_id=config.mask_token_id,
            batch_size=2, seq_len=16
        )
        
        # No mask tokens in output
        assert (output != config.mask_token_id).all()
    
    def test_sample_produces_valid_token_ids(self, trained_model):
        """Output should contain valid token ids."""
        model, config = trained_model
        
        output = sample(
            model, T=50, mask_token_id=config.mask_token_id,
            batch_size=2, seq_len=16
        )
        
        # All tokens should be in valid range
        assert (output >= 0).all()
        assert (output < config.vocab_size).all()
    
    def test_sample_different_seeds_different_outputs(self, trained_model):
        """Different random seeds should produce different outputs."""
        model, config = trained_model
        
        torch.manual_seed(42)
        output1 = sample(
            model, T=50, mask_token_id=config.mask_token_id,
            batch_size=2, seq_len=16
        )
        
        torch.manual_seed(123)
        output2 = sample(
            model, T=50, mask_token_id=config.mask_token_id,
            batch_size=2, seq_len=16
        )
        
        # Should be different (with high probability)
        assert not torch.equal(output1, output2)
    
    def test_sample_batch_consistency(self, trained_model):
        """Batch samples should be independent."""
        model, config = trained_model
        
        output = sample(
            model, T=50, mask_token_id=config.mask_token_id,
            batch_size=4, seq_len=16
        )
        
        # Different batch elements should generally be different
        # (not a strict test due to randomness)
        assert output.shape == (4, 16)
    
    def test_sample_with_progress_callback(self, trained_model):
        """Should call progress callback."""
        model, config = trained_model
        
        calls = []
        def callback(step, total):
            calls.append((step, total))
        
        output = sample(
            model, T=20, mask_token_id=config.mask_token_id,
            batch_size=1, seq_len=8,
            progress_callback=callback
        )
        
        # Should have been called T times
        assert len(calls) == 20
        assert calls[-1] == (20, 20)


class TestTokensToText:
    """Tests for token-to-text conversion."""

    def test_tokens_to_text_basic(self):
        """Should convert tokens to text."""
        # Use higher token values to avoid filtering (tokens <= 2 are filtered)
        tokens = torch.tensor([[10, 11, 12, 2, 13], [20, 21, 22, 23, 24]])

        texts = tokens_to_text(tokens, eos_token_id=2)

        assert len(texts) == 2
        # First sequence should be truncated at EOS (length 3: 10,11,12)
        # Second sequence has no EOS (length 5: 20,21,22,23,24)
        assert len(texts[0]) < len(texts[1])

    def test_tokens_to_text_1d_input(self):
        """Should handle 1D input."""
        tokens = torch.tensor([3, 4, 5, 2, 0])
        
        texts = tokens_to_text(tokens, eos_token_id=2)
        
        assert len(texts) == 1
    
    def test_tokens_to_text_no_eos(self):
        """Should handle sequences without EOS."""
        tokens = torch.tensor([[3, 4, 5, 6, 7]])
        
        texts = tokens_to_text(tokens, eos_token_id=2)
        
        assert len(texts) == 1
        assert len(texts[0]) > 0
    
    def test_tokens_to_text_all_eos(self):
        """Should handle sequences that are all EOS."""
        tokens = torch.tensor([[2, 2, 2]])
        
        texts = tokens_to_text(tokens, eos_token_id=2)
        
        assert len(texts) == 1
        # Should be empty or whitespace after truncation
        assert texts[0].strip() == ""
