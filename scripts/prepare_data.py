#!/usr/bin/env python3
"""
Data preparation script for discrete diffusion language model.

Prepares text data by training a tokenizer and creating tokenized datasets.

Usage:
    python scripts/prepare_data.py --input data/raw.txt --output data/processed
    python scripts/prepare_data.py --input data/ --output data/processed --vocab-size 32000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusionllm.tokenizer import DiffusionTokenizer


def find_text_files(input_path: str) -> List[str]:
    """Find all text files in directory or return single file."""
    path = Path(input_path)
    
    if path.is_file():
        return [str(path)]
    
    if path.is_dir():
        # Find common text file extensions
        extensions = {".txt", ".md", ".json", ".jsonl"}
        files = []
        for ext in extensions:
            files.extend(str(p) for p in path.rglob(f"*{ext}"))
        return sorted(files)
    
    raise ValueError(f"Input path not found: {input_path}")


def load_text_from_files(files: List[str]) -> str:
    """Load and concatenate text from files."""
    texts = []
    
    for file_path in files:
        print(f"  Reading: {file_path}")
        
        if file_path.endswith(".json") or file_path.endswith(".jsonl"):
            # JSON format - extract text from 'text' field
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".jsonl"):
                    for line in f:
                        data = json.loads(line)
                        if "text" in data:
                            texts.append(data["text"])
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if "text" in item:
                                texts.append(item["text"])
                    elif "text" in data:
                        texts.append(data["text"])
        else:
            # Plain text
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    
    return "\n".join(texts)


def prepare_data(
    input_path: str,
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    train_split: float = 0.9,
    val_split: float = 0.05,
    test_split: float = 0.05,
    seq_len: int = 512,
    overwrite: bool = False,
):
    """
    Prepare data for training with text packing.
    
    Text packing:
    1. Tokenize entire corpus
    2. Concatenate all tokens
    3. Chunk into fixed-length sequences (seq_len)
    
    This ensures all training samples are exactly seq_len tokens,
    maximizing training efficiency.
    
    Args:
        input_path: Input file or directory
        output_dir: Output directory for processed data
        vocab_size: Vocabulary size
        min_frequency: Minimum token frequency
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        seq_len: Fixed sequence length for training
        overwrite: Overwrite existing files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    tokenizer_path = output_path / "tokenizer.json"
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"
    test_path = output_path / "test.jsonl"
    
    if not overwrite and all(p.exists() for p in [tokenizer_path, train_path]):
        print(f"Data already processed at {output_dir}")
        print("Use --overwrite to regenerate")
        return
    
    # Find input files
    print(f"\n{'='*60}")
    print("Finding input files...")
    files = find_text_files(input_path)
    print(f"Found {len(files)} file(s)")
    
    # Load text
    print(f"\n{'='*60}")
    print("Loading text data...")
    text = load_text_from_files(files)
    print(f"Loaded {len(text):,} characters")
    
    # Split into train/val/test
    print(f"\n{'='*60}")
    print("Splitting data...")
    
    # Split by lines to maintain document boundaries
    lines = text.split("\n")
    n_lines = len(lines)
    
    train_end = int(n_lines * train_split)
    val_end = int(n_lines * (train_split + val_split))
    
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    train_text = "\n".join(train_lines)
    val_text = "\n".join(val_lines)
    test_text = "\n".join(test_lines)
    
    print(f"  Train: {len(train_lines):,} lines ({len(train_text):,} chars)")
    print(f"  Val:   {len(val_lines):,} lines ({len(val_text):,} chars)")
    print(f"  Test:  {len(test_lines):,} lines ({len(test_text):,} chars)")
    
    # Train tokenizer
    print(f"\n{'='*60}")
    print("Training tokenizer...")
    
    # Create temp file for training
    temp_train_path = output_path / "train_raw.txt"
    with open(temp_train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    
    tokenizer = DiffusionTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    tokenizer.train(str(temp_train_path), show_progress=True)
    tokenizer.save(str(tokenizer_path))
    
    # Clean up temp file
    temp_train_path.unlink()
    
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.actual_vocab_size}")

    # Tokenize and pack text into fixed-length sequences
    print(f"\n{'='*60}")
    print(f"Packing text into {seq_len}-token sequences...")

    def tokenize_and_pack(text: str) -> list:
        """Tokenize text and pack into fixed-length sequences."""
        # Tokenize entire text
        all_tokens = tokenizer.encode(text, add_bos=True, add_eos=False)
        
        # Chunk into fixed-length sequences
        sequences = []
        for i in range(0, len(all_tokens) - seq_len + 1, seq_len):
            seq = all_tokens[i:i + seq_len]
            if len(seq) == seq_len:
                sequences.append(seq)
        
        return sequences

    # Process each split
    print("Tokenizing train split...")
    train_sequences = tokenize_and_pack(train_text)
    print(f"  Train: {len(train_sequences):,} sequences ({len(train_sequences) * seq_len:,} tokens)")

    print("Tokenizing val split...")
    val_sequences = tokenize_and_pack(val_text)
    print(f"  Val: {len(val_sequences):,} sequences ({len(val_sequences) * seq_len:,} tokens)")

    print("Tokenizing test split...")
    test_sequences = tokenize_and_pack(test_text)
    print(f"  Test: {len(test_sequences):,} sequences ({len(test_sequences) * seq_len:,} tokens)")

    # Save sequences
    def save_sequences(sequences: list, output_file: Path, tokenizer_obj, description: str):
        with open(output_file, "w", encoding="utf-8") as f:
            for i, seq in enumerate(sequences):
                text = tokenizer_obj.decode(seq, skip_special_tokens=False)
                record = {
                    "token_ids": seq,
                    "length": len(seq),
                    "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for storage
                }
                f.write(json.dumps(record) + "\n")
        print(f"  {description}: Saved {len(sequences):,} sequences to {output_file}")

    save_sequences(train_sequences, train_path, tokenizer, "Train")
    save_sequences(val_sequences, val_path, tokenizer, "Val")
    save_sequences(test_sequences, test_path, tokenizer, "Test")
    
    # Save metadata
    metadata = {
        "vocab_size": tokenizer.actual_vocab_size,
        "seq_len": seq_len,
        "max_seq_len": seq_len,  # All sequences are exactly seq_len
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "test_sequences": len(test_sequences),
        "train_tokens": len(train_sequences) * seq_len,
        "val_tokens": len(val_sequences) * seq_len,
        "test_tokens": len(test_sequences) * seq_len,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "test_chars": len(test_text),
        "min_frequency": min_frequency,
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"\nOutput files:")
    print(f"  Tokenizer:  {tokenizer_path}")
    print(f"  Train:      {train_path}")
    print(f"  Val:        {val_path}")
    print(f"  Test:       {test_path}")
    print(f"  Metadata:   {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for diffusion LM training")
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input file or directory containing text data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for BPE (default: 2)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Training set proportion (default: 0.9)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Validation set proportion (default: 0.05)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.05,
        help="Test set proportion (default: 0.05)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed data"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for training (default: 512). Text will be packed to this length."
    )
    
    args = parser.parse_args()
    
    # Validate splits
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 0.001:
        print(f"Error: Splits must sum to 1.0, got {total_split}")
        sys.exit(1)
    
    prepare_data(
        input_path=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
