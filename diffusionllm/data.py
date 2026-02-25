"""
Dataset classes for discrete diffusion language model.

Efficient loading of tokenized datasets with streaming support.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class TokenizedDataset(Dataset):
    """
    Dataset for loading tokenized text data.
    
    Loads pre-tokenized data from JSONL files created by prepare_data.py.
    """
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        pad_token_id: int = 1,
        mask_token_id: int = 0,
        cache_in_memory: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL file or directory with train/val/test.jsonl
            max_length: Maximum sequence length (sequences are truncated/padded)
            pad_token_id: Padding token ID
            mask_token_id: Mask token ID
            cache_in_memory: Load all data into memory (faster but uses more RAM)
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        # Find data file
        if self.data_path.is_dir():
            self.data_path = self.data_path / "train.jsonl"
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load data
        self.cache_in_memory = cache_in_memory
        self.data: List[Dict] = []
        self.indices: List[int] = []
        self.line_offsets: List[int] = []
        
        if cache_in_memory:
            self._load_all_data()
        else:
            self._build_index()
    
    def _load_all_data(self):
        """Load all data into memory."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.indices = list(range(len(self.data)))
    
    def _build_index(self):
        """Build index of line offsets for lazy loading."""
        self.line_offsets = [0]
        current_offset = 0
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                current_offset += len(line.encode('utf-8'))
                self.line_offsets.append(current_offset)
        
        self.indices = list(range(len(self.line_offsets) - 1))
    
    def _get_item_lazy(self, idx: int) -> Dict:
        """Get item by lazy loading from disk."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            return json.loads(line)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data item."""
        if self.cache_in_memory:
            item = self.data[idx]
        else:
            item = self._get_item_lazy(idx)
        
        # Get token IDs
        token_ids = item.get("token_ids", [])
        
        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_token_id] * (self.max_length - len(token_ids))
        
        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "length": item.get("length", len(token_ids)),
        }


class StreamingDataset(Dataset):
    """
    Streaming dataset for very large files.
    
    Reads data on-the-fly without building a full index.
    Suitable for datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        pad_token_id: int = 1,
        mask_token_id: int = 0,
        buffer_size: int = 10000,
        seed: int = 42,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_path: Path to JSONL file
            max_length: Maximum sequence length
            pad_token_id: Padding token ID
            mask_token_id: Mask token ID
            buffer_size: Number of lines to buffer for shuffling
            seed: Random seed for shuffling
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.buffer_size = buffer_size
        self.seed = seed
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Count lines
        self.num_lines = self._count_lines()
    
    def _count_lines(self) -> int:
        """Count number of lines in file."""
        count = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count
    
    def __len__(self) -> int:
        return self.num_lines
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by seeking to line (slow but memory efficient)."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    item = json.loads(line)
                    break
        
        token_ids = item.get("token_ids", [])
        
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_token_id] * (self.max_length - len(token_ids))
        
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "length": item.get("length", len(token_ids)),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with optimal settings.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete batches
    
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def load_datasets(
    data_dir: str,
    max_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_in_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Load train, validation, and test datasets.
    
    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers per dataset
        cache_in_memory: Cache data in memory
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    data_path = Path(data_dir)
    
    datasets = {}
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_path = data_path / f"{split}.jsonl"
        if split_path.exists():
            dataset = TokenizedDataset(
                str(split_path),
                max_length=max_length,
                cache_in_memory=cache_in_memory if split == "train" else True,
            )
            
            datasets[split] = create_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                drop_last=(split == "train"),
            )
            print(f"Loaded {split}: {len(dataset):,} samples")
        else:
            print(f"Warning: {split_path} not found")
    
    return datasets
