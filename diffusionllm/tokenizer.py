"""
Tokenizer module for discrete diffusion language model.

Supports BPE tokenizer via Hugging Face tokenizers library.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer, trainers, pre_tokenizers, normalizers
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing


class DiffusionTokenizer:
    """
    Wrapper around Hugging Face tokenizer for diffusion models.
    
    Handles special tokens (MASK, PAD, EOS, BOS) and provides
    a simple encode/decode interface.
    """
    
    # Special tokens
    MASK_TOKEN = "[MASK]"
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "[EOS]"
    BOS_TOKEN = "[BOS]"
    UNK_TOKEN = "[UNK]"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum token frequency for BPE merges
            special_tokens: Additional special tokens
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Define special tokens with their IDs
        self.special_tokens = special_tokens or []
        self.all_special_tokens = [
            self.MASK_TOKEN,
            self.PAD_TOKEN,
            self.EOS_TOKEN,
            self.BOS_TOKEN,
            self.UNK_TOKEN,
        ] + self.special_tokens
        
        # Token ID mappings
        self._special_token_to_id = {
            tok: idx for idx, tok in enumerate(self.all_special_tokens)
        }
        self._id_to_special_token = {
            idx: tok for tok, idx in self._special_token_to_id.items()
        }
        
        # Number of special tokens
        self.num_special_tokens = len(self.all_special_tokens)
        
        # Initialize the tokenizer
        self._tokenizer = None
        self._is_trained = False
    
    @property
    def tokenizer(self) -> Tokenizer:
        """Get the underlying tokenizer."""
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer not initialized. Call train() or load() first."
            )
        return self._tokenizer
    
    @property
    def is_trained(self) -> bool:
        """Check if tokenizer is trained."""
        return self._is_trained
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including special tokens)."""
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, value: int):
        """Set vocabulary size (excluding special tokens)."""
        self._vocab_size = value
    
    @property
    def actual_vocab_size(self) -> int:
        """Get actual vocabulary size (including special tokens)."""
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return self.vocab_size + self.num_special_tokens
    
    def train(self, files: Union[str, List[str]], show_progress: bool = True):
        """
        Train the tokenizer on text files.
        
        Args:
            files: Path to file or list of files to train on
            show_progress: Show training progress
        """
        if isinstance(files, str):
            files = [files]
        
        # Initialize BPE tokenizer
        self._tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        
        # Configure pre-tokenization (split on whitespace and punctuation)
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Configure normalization
        self._tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC(),
            normalizers.Lowercase(),
        ])
        
        # Create trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size - self.num_special_tokens,
            min_frequency=self.min_frequency,
            special_tokens=self.all_special_tokens,
            show_progress=show_progress,
        )
        
        # Train
        self._tokenizer.train(files, trainer=trainer)
        
        # Set post-processor (no special template, just tokens)
        self._tokenizer.post_processor = TemplateProcessing(
            single="$A",
            special_tokens=[
                (self.MASK_TOKEN, self.mask_token_id),
                (self.PAD_TOKEN, self.pad_token_id),
                (self.EOS_TOKEN, self.eos_token_id),
                (self.BOS_TOKEN, self.bos_token_id),
            ],
        )
        
        self._is_trained = True
    
    def save(self, path: str):
        """Save tokenizer to file."""
        if not self._is_trained:
            raise ValueError("Cannot save untrained tokenizer")
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)
    
    @classmethod
    def load(cls, path: str) -> "DiffusionTokenizer":
        """Load tokenizer from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        
        # Create instance
        tokenizer = cls()
        tokenizer._tokenizer = Tokenizer.from_file(path)
        tokenizer._is_trained = True
        
        # Update vocab size from loaded tokenizer
        tokenizer._vocab_size = tokenizer.tokenizer.get_vocab_size()
        
        # Rebuild special token mappings from loaded tokenizer
        vocab = tokenizer.tokenizer.get_vocab()
        tokenizer._special_token_to_id = {}
        tokenizer._id_to_special_token = {}
        
        for tok in tokenizer.all_special_tokens:
            if tok in vocab:
                token_id = vocab[tok]
                tokenizer._special_token_to_id[tok] = token_id
                tokenizer._id_to_special_token[token_id] = tok
        
        tokenizer.num_special_tokens = len(tokenizer._special_token_to_id)
        
        return tokenizer
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add BOS token
            add_eos: Add EOS token
            truncation: Truncate to max_length
            max_length: Maximum sequence length
        
        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        
        # Add special tokens
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        
        # Truncate
        if truncation and max_length is not None:
            ids = ids[:max_length]
        
        return ids
    
    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        pad_to_multiple_of: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_bos: Add BOS token
            add_eos: Add EOS token
            truncation: Truncate to max_length
            max_length: Maximum sequence length
            padding: Pad to same length
            pad_to_multiple_of: Pad to multiple of this value
        
        Returns:
            List of token ID lists
        """
        encodings = self.tokenizer.encode_batch(texts)
        
        # Process each encoding
        all_ids = []
        max_len = 0
        
        for encoding in encodings:
            ids = encoding.ids
            
            if add_bos:
                ids = [self.bos_token_id] + ids
            if add_eos:
                ids = ids + [self.eos_token_id]
            
            if truncation and max_length is not None:
                ids = ids[:max_length]
            
            all_ids.append(ids)
            max_len = max(max_len, len(ids))
        
        # Pad if requested
        if padding:
            if pad_to_multiple_of is not None:
                max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            
            for i, ids in enumerate(all_ids):
                padding_len = max_len - len(ids)
                if padding_len > 0:
                    all_ids[i] = ids + [self.pad_token_id] * padding_len
        
        return all_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out special tokens
            token_ids = [
                tid for tid in token_ids 
                if tid not in self._id_to_special_token
            ]
        
        return self.tokenizer.decode(token_ids)
    
    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID lists."""
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_token_ids
        ]
    
    # Special token ID properties
    @property
    def mask_token_id(self) -> int:
        """Get MASK token ID."""
        return self._special_token_to_id.get(self.MASK_TOKEN, 0)
    
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self._special_token_to_id.get(self.PAD_TOKEN, 1)
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self._special_token_to_id.get(self.EOS_TOKEN, 2)
    
    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self._special_token_to_id.get(self.BOS_TOKEN, 3)
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self._special_token_to_id.get(self.UNK_TOKEN, 4)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.actual_vocab_size
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_bos: bool = False,
        add_eos: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
    ):
        """Callable interface for encoding."""
        if isinstance(text, str):
            return self.encode(
                text, add_bos=add_bos, add_eos=add_eos,
                truncation=truncation, max_length=max_length
            )
        else:
            return self.encode_batch(
                text, add_bos=add_bos, add_eos=add_eos,
                truncation=truncation, max_length=max_length,
                padding=padding
            )
