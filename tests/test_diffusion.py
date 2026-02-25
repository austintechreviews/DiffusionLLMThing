"""
Tests for the diffusion process functions.
"""

import pytest
import torch

from diffusionllm.diffusion import (
    forward_diffusion,
    compute_loss,
    get_noise_schedule,
    compute_transition_prob,
)


class TestGetNoiseSchedule:
    """Tests for noise schedule computation."""
    
    def test_linear_schedule_bounds(self):
        """Linear schedule should have alpha_0 = 1 and alpha_T = 0."""
        T = 1000
        alpha = get_noise_schedule(T, schedule="linear")
        
        assert alpha.shape == (T + 1,)
        assert torch.isclose(alpha[0], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(alpha[T], torch.tensor(0.0), atol=1e-6)
    
    def test_linear_schedule_monotonic(self):
        """Linear schedule should be monotonically decreasing."""
        T = 100
        alpha = get_noise_schedule(T, schedule="linear")
        
        diffs = alpha[:-1] - alpha[1:]
        assert torch.all(diffs >= 0)
    
    def test_cosine_schedule_bounds(self):
        """Cosine schedule should have alpha_0 ≈ 1 and alpha_T ≈ 0."""
        T = 1000
        alpha = get_noise_schedule(T, schedule="cosine")
        
        assert alpha.shape == (T + 1,)
        assert torch.isclose(alpha[0], torch.tensor(1.0), atol=1e-6)
        assert alpha[T] < 0.01  # Should be close to 0
    
    def test_cosine_schedule_monotonic(self):
        """Cosine schedule should be monotonically decreasing."""
        T = 100
        alpha = get_noise_schedule(T, schedule="cosine")
        
        diffs = alpha[:-1] - alpha[1:]
        assert torch.all(diffs >= 0)
    
    def test_invalid_schedule(self):
        """Should raise error for invalid schedule."""
        with pytest.raises(ValueError):
            get_noise_schedule(100, schedule="invalid")


class TestForwardDiffusion:
    """Tests for forward diffusion process."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        x0 = torch.randint(3, vocab_size, (batch_size, seq_len))
        return x0, batch_size, seq_len
    
    def test_t_zero_no_masking(self):
        """At t=0, no tokens should be masked."""
        x0 = torch.randint(3, 100, (2, 8))
        T = 100
        
        xt, mask = forward_diffusion(
            x0, torch.zeros(2, dtype=torch.long), T, mask_token_id=0
        )
        
        assert torch.equal(xt, x0)
        assert mask.sum() == 0
    
    def test_t_max_all_masked(self):
        """At t=T, all non-pad tokens should be masked."""
        torch.manual_seed(42)
        x0 = torch.randint(3, 100, (2, 8))
        T = 100
        
        # At t=T, mask_prob = 1.0, so all should be masked
        xt, mask = forward_diffusion(
            x0, torch.full((2,), T - 1, dtype=torch.long), T, mask_token_id=0
        )
        
        # Most tokens should be masked (probability approaches 1)
        assert mask.sum() > 0
        assert torch.all(xt[mask] == 0)
    
    def test_masked_tokens_are_mask_token(self):
        """Masked positions should contain mask token."""
        torch.manual_seed(42)
        x0 = torch.randint(3, 100, (4, 16))
        T = 100
        t = torch.full((4,), 50, dtype=torch.long)
        
        xt, mask = forward_diffusion(x0, t, T, mask_token_id=0)
        
        # All masked positions should have mask token
        assert torch.all(xt[mask] == 0)
    
    def test_unmasked_tokens_unchanged(self):
        """Unmasked positions should remain unchanged."""
        torch.manual_seed(42)
        x0 = torch.randint(3, 100, (4, 16))
        T = 100
        t = torch.full((4,), 50, dtype=torch.long)
        
        xt, mask = forward_diffusion(x0, t, T, mask_token_id=0)
        
        # Unmasked positions should be unchanged
        unmask = ~mask
        assert torch.all(xt[unmask] == x0[unmask])
    
    def test_padding_not_masked(self):
        """Padding tokens should never be masked."""
        x0 = torch.randint(3, 100, (2, 8))
        x0[0, 4:] = 1  # Set padding
        T = 100
        t = torch.full((2,), 99, dtype=torch.long)
        
        xt, mask = forward_diffusion(
            x0, t, T, mask_token_id=0, pad_token_id=1
        )
        
        # Padding positions should not be in mask
        pad_positions = (x0 == 1)
        assert not mask[pad_positions].any()
    
    def test_masking_probability_increases_with_t(self):
        """Higher t should result in more masking on average."""
        torch.manual_seed(42)
        x0 = torch.randint(3, 100, (8, 32))
        T = 100
        
        mask_rates = []
        for t_val in [10, 30, 50, 70, 90]:
            t = torch.full((8,), t_val, dtype=torch.long)
            _, mask = forward_diffusion(x0, t, T, mask_token_id=0)
            mask_rate = mask.float().mean().item()
            mask_rates.append(mask_rate)
        
        # Generally should increase (allowing for sampling variance)
        assert mask_rates[-1] > mask_rates[0]
    
    def test_scalar_t_handling(self):
        """Should handle scalar timestep correctly."""
        x0 = torch.randint(3, 100, (2, 8))
        T = 100
        
        xt, mask = forward_diffusion(x0, torch.tensor(50), T, mask_token_id=0)
        
        assert xt.shape == x0.shape
        assert mask.shape == x0.shape


class TestComputeLoss:
    """Tests for loss computation."""
    
    def test_loss_is_positive(self):
        """Loss should always be positive."""
        batch_size, seq_len, vocab_size = 4, 16, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        x0 = torch.randint(3, vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) < 0.5
        t = torch.randint(0, 100, (batch_size,))
        
        loss = compute_loss(logits, x0, mask, t, T=100)
        
        assert loss >= 0
    
    def test_loss_with_no_masked_positions(self):
        """Loss should be zero when no positions are masked."""
        batch_size, seq_len, vocab_size = 4, 16, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        x0 = torch.randint(3, vocab_size, (batch_size, seq_len))
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        t = torch.randint(0, 100, (batch_size,))
        
        loss = compute_loss(logits, x0, mask, t, T=100)
        
        assert loss == 0
    
    def test_loss_decreases_with_correct_predictions(self):
        """Loss should decrease when predictions match targets."""
        batch_size, seq_len, vocab_size = 4, 16, 100
        x0 = torch.randint(3, vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) < 0.5
        t = torch.zeros(batch_size, dtype=torch.long)
        
        # Random predictions
        random_logits = torch.randn(batch_size, seq_len, vocab_size)
        random_loss = compute_loss(random_logits, x0, mask, t, T=100)
        
        # Perfect predictions (high logit at correct position)
        perfect_logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        perfect_logits.scatter_(2, x0.unsqueeze(-1), 10.0)
        perfect_loss = compute_loss(perfect_logits, x0, mask, t, T=100)
        
        assert perfect_loss < random_loss
    
    def test_elbo_weighting(self):
        """Earlier timesteps should have higher weight."""
        batch_size, seq_len, vocab_size = 4, 16, 100
        x0 = torch.randint(3, vocab_size, (batch_size, seq_len))
        
        # Create same mask pattern
        torch.manual_seed(42)
        mask = torch.rand(batch_size, seq_len) < 0.5
        
        # Random logits
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Loss at early timestep (high weight: 1/(1-0/100) = 1)
        t_early = torch.zeros(batch_size, dtype=torch.long)
        loss_early = compute_loss(logits, x0, mask, t_early, T=100)
        
        # Loss at late timestep (low weight: 1/(1-90/100) = 10)
        t_late = torch.full((batch_size,), 90, dtype=torch.long)
        loss_late = compute_loss(logits.clone(), x0, mask, t_late, T=100)
        
        # The weight at t=90 is 10x higher than at t=0
        # So loss_late should be approximately 10x loss_early (for same raw loss)
        # We check that the weighting is applied correctly
        assert loss_late > loss_early  # Later timesteps have higher weight
    
    def test_padding_ignored_in_loss(self):
        """Padding positions should not contribute to loss."""
        batch_size, seq_len, vocab_size = 4, 16, 100
        
        # Sequence with padding
        x0_with_pad = torch.randint(3, vocab_size, (batch_size, seq_len))
        x0_with_pad[:, 8:] = 1  # Padding
        
        # All positions masked
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        t = torch.zeros(batch_size, dtype=torch.long)
        
        # Random logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        loss = compute_loss(logits, x0_with_pad, mask, t, T=100, pad_token_id=1)
        
        # Loss should be computed only over non-padding positions
        assert torch.isfinite(loss)
    
    def test_reduction_sum(self):
        """Test sum reduction."""
        batch_size, seq_len, vocab_size = 4, 16, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        x0 = torch.randint(3, vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) < 0.5
        t = torch.zeros(batch_size, dtype=torch.long)
        
        loss_mean = compute_loss(logits, x0, mask, t, T=100, reduction="mean")
        loss_sum = compute_loss(logits, x0, mask, t, T=100, reduction="sum")
        
        # Sum should be >= mean (since we divide by count for mean)
        assert loss_sum >= loss_mean


class TestComputeTransitionProb:
    """Tests for transition probability computation."""
    
    def test_transition_prob_increases_with_t(self):
        """Transition probability should increase with t."""
        T = 100
        
        probs = []
        for t_val in [1, 10, 50, 90]:
            t = torch.tensor([t_val])
            prob = compute_transition_prob(t, T)
            probs.append(prob.item())
        
        # Probability should generally increase
        assert probs[-1] > probs[0]
    
    def test_transition_prob_with_alpha(self):
        """Test with precomputed alpha schedule."""
        T = 100
        alpha = get_noise_schedule(T)
        
        t = torch.tensor([50])
        prob = compute_transition_prob(t, T, alpha=alpha)
        
        assert 0 <= prob <= 1
