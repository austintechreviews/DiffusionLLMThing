"""
Tests for training utilities.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from diffusionllm.utils import (
    get_lr_schedule,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    TrainingLogger,
)


class TestGetLrSchedule:
    """Tests for learning rate schedule."""
    
    def test_warmup_phase(self):
        """LR should increase linearly during warmup."""
        warmup_steps = 1000
        
        lr1 = get_lr_schedule(0, warmup_steps, 10000)
        lr2 = get_lr_schedule(500, warmup_steps, 10000)
        lr3 = get_lr_schedule(1000, warmup_steps, 10000)
        
        assert lr1 == 0.0
        assert lr2 == 0.5
        assert lr3 == 1.0
    
    def test_decay_phase(self):
        """LR should decrease during decay phase."""
        warmup_steps = 1000
        max_steps = 10000
        
        lr_start = get_lr_schedule(warmup_steps, warmup_steps, max_steps)
        lr_mid = get_lr_schedule(5500, warmup_steps, max_steps)
        lr_end = get_lr_schedule(max_steps, warmup_steps, max_steps)
        
        assert lr_start == 1.0
        assert lr_mid < lr_start
        assert lr_end < lr_mid
        assert lr_end == 0.0  # Cosine decay reaches 0
    
    def test_cosine_shape(self):
        """LR schedule should follow cosine curve."""
        warmup_steps = 100
        max_steps = 1000
        
        lrs = [get_lr_schedule(s, warmup_steps, max_steps) for s in range(max_steps + 1)]
        
        # After warmup, should be monotonically decreasing
        for i in range(warmup_steps + 1, len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]


class TestCountParameters:
    """Tests for parameter counting."""
    
    def test_simple_module(self):
        """Should correctly count parameters."""
        model = nn.Linear(10, 5)
        
        count = count_parameters(model)
        
        # weight (10*5) + bias (5) = 55
        assert count == 55
    
    def test_nested_module(self):
        """Should count parameters in nested modules."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        )
        
        count = count_parameters(model)
        
        # (10*5 + 5) + (5*2 + 2) = 55 + 12 = 67
        assert count == 67
    
    def test_excludes_non_trainable(self):
        """Should only count trainable parameters."""
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False
        
        count = count_parameters(model)
        
        # Only bias should be counted
        assert count == 5


class TestCheckpoint:
    """Tests for checkpoint save/load."""
    
    def test_save_and_load_checkpoint(self):
        """Should save and load checkpoint correctly."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        config = {'lr': 0.001, 'batch_size': 32}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            
            # Save
            save_checkpoint(100, model, optimizer, scaler, config, path)
            
            # Modify model
            with torch.no_grad():
                model.weight.fill_(999)
            
            # Load
            _, loaded_model, loaded_optimizer, loaded_scaler = load_checkpoint(
                path, model, optimizer, scaler
            )
            
            # Check model weights are restored
            assert not torch.all(model.weight == 999)
    
    def test_load_returns_step(self):
        """Should return the saved step."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        config = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            
            save_checkpoint(500, model, optimizer, scaler, config, path)
            
            step, _, _, _ = load_checkpoint(path, model)
            
            assert step == 500
    
    def test_load_without_optimizer(self):
        """Should load without optimizer state."""
        model = nn.Linear(10, 5)
        config = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            
            optimizer = torch.optim.Adam(model.parameters())
            scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
            
            save_checkpoint(100, model, optimizer, scaler, config, path)
            
            # Load without optimizer
            _, loaded_model, loaded_optimizer, loaded_scaler = load_checkpoint(
                path, model, optimizer=None, scaler=None, load_optimizer=False
            )
            
            assert loaded_optimizer is None
            assert loaded_scaler is None
    
    def test_checkpoint_creates_directory(self):
        """Should create checkpoint directory if needed."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        config = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'subdir', 'checkpoint.pt')
            
            save_checkpoint(100, model, optimizer, scaler, config, path)
            
            assert os.path.exists(path)


class TestAverageMeter:
    """Tests for AverageMeter utility."""
    
    def test_initial_values(self):
        """Should initialize to zero."""
        meter = AverageMeter("test")
        
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0
    
    def test_single_update(self):
        """Should correctly update with single value."""
        meter = AverageMeter()
        meter.update(5.0)
        
        assert meter.val == 5.0
        assert meter.avg == 5.0
        assert meter.sum == 5.0
        assert meter.count == 1
    
    def test_multiple_updates(self):
        """Should correctly average multiple values."""
        meter = AverageMeter()
        meter.update(2.0, n=1)
        meter.update(4.0, n=1)
        meter.update(6.0, n=1)
        
        assert meter.val == 6.0
        assert meter.avg == 4.0  # (2+4+6)/3
        assert meter.sum == 12.0
        assert meter.count == 3
    
    def test_batched_update(self):
        """Should handle batched updates."""
        meter = AverageMeter()
        meter.update(5.0, n=4)  # Batch of 4
        
        assert meter.val == 5.0
        assert meter.sum == 20.0
        assert meter.count == 4
    
    def test_string_representation(self):
        """Should have informative string representation."""
        meter = AverageMeter("loss")
        meter.update(0.1234)
        
        str_repr = str(meter)
        
        assert "loss" in str_repr
        assert "0.1234" in str_repr


class TestTrainingLogger:
    """Tests for TrainingLogger."""
    
    def test_logger_creation(self):
        """Should create logger without errors."""
        logger = TrainingLogger()
        assert logger is not None
    
    def test_logger_logs_metrics(self, capsys):
        """Should log metrics to console."""
        logger = TrainingLogger()
        logger.log({'loss': 0.5, 'ppl': 10.0}, step=100)
        
        captured = capsys.readouterr()
        assert "Step" in captured.out
        assert "loss" in captured.out
        assert "0.5" in captured.out
    
    def test_logger_closes(self):
        """Should close without errors."""
        logger = TrainingLogger()
        logger.close()  # Should not raise
