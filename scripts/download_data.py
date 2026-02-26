#!/usr/bin/env python3
"""
Dataset downloader for discrete diffusion language model.

Downloads and prepares popular text datasets for training.

Usage:
    python scripts/download_data.py --dataset shakespeare --output data/raw/shakespeare
    python scripts/download_data.py --dataset tinyshakespeare --output data/raw/
    python scripts/download_data.py --dataset wikitext-2 --output data/raw/wikitext2
"""

import argparse
import gzip
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Dataset configurations
DATASETS = {
    # Small datasets for testing
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "shakespeare.txt",
        "description": "Tiny Shakespeare (1MB) - character-level text",
        "size": "~1MB",
    },
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "tinyshakespeare.txt",
        "description": "Tiny Shakespeare (1MB)",
        "size": "~1MB",
    },
    
    # Medium datasets
    "wikitext-2": {
        "url": "https://raw.githubusercontent.com/datasets/wikitext/master/wikitext-2-raw.zip",
        "filename": "wikitext-2-raw.zip",
        "extract": True,
        "description": "WikiText-2 (2MB) - Wikipedia articles",
        "size": "~2MB",
        "note": "Alternative: pip install datasets; from datasets import load_dataset; ds = load_dataset('wikitext', 'wikitext-2-raw-v1')",
    },
    "wikitext-103": {
        "url": "https://raw.githubusercontent.com/datasets/wikitext/master/wikitext-103-raw.zip",
        "filename": "wikitext-103-raw.zip",
        "extract": True,
        "description": "WikiText-103 (600MB) - Larger Wikipedia corpus",
        "size": "~600MB",
        "note": "Alternative: pip install datasets; from datasets import load_dataset; ds = load_dataset('wikitext', 'wikitext-103-raw-v1')",
    },
    
    # Large datasets
    "enwik8": {
        "url": "http://mattmahoney.net/dc/enwik8.zip",
        "filename": "enwik8.zip",
        "extract": True,
        "description": "enwik8 (100MB) - First 100MB of Wikipedia XML dump",
        "size": "~100MB",
    },
    "enwik9": {
        "url": "http://mattmahoney.net/dc/enwik9.zip",
        "filename": "enwik9.zip",
        "extract": True,
        "description": "enwik9 (1GB) - First 1GB of Wikipedia XML dump",
        "size": "~1GB",
    },
    
    # Books and literature
    "pg19-sample": {
        "url": "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/train-00000-of-00001.parquet?download=true",
        "filename": "pg19_sample.parquet",
        "description": "Project Gutenberg books sample (requires parquet)",
        "size": "~100MB",
        "note": "Requires: pip install pandas pyarrow",
    },
    
    # Code datasets
    "python-algo": {
        "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/README.md",
        "filename": "python_algos.txt",
        "description": "Python Algorithms (sample from TheAlgorithms/Python)",
        "size": "~10MB",
        "note": "Full repo: git clone https://github.com/TheAlgorithms/Python",
    },
    
    # News/Articles
    "ag-news": {
        "url": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "filename": "ag_news.csv",
        "description": "AG News headlines (sample)",
        "size": "~10MB",
    },
}


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    if not HAS_REQUESTS:
        print("Error: 'requests' library required. Install with: pip install requests")
        return False
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directory
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            if HAS_TQDM:
                progress = tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=destination.name
                )
            else:
                progress = None
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if progress:
                        progress.update(len(chunk))
            
            if progress:
                progress.close()
        
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip/tar.gz archive."""
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.suffix in ['.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_dir)
        elif archive_path.suffix == '.tar':
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"Extracted to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def extract_text_from_wikitext(zip_path: Path, output_dir: Path) -> Path:
    """Extract train.txt from WikiText zip."""
    extract_archive(zip_path, output_dir)
    
    # Find the extracted txt file
    for txt_file in output_dir.rglob("*.txt"):
        if "train" in txt_file.name.lower():
            # Move to output_dir root
            dest = output_dir / "train.txt"
            shutil.move(str(txt_file), str(dest))
            return dest
    
    return None


def download_shakespeare(output_dir: Path) -> bool:
    """Download Tiny Shakespeare dataset."""
    output_file = output_dir / "shakespeare.txt"
    return download_file(DATASETS["shakespeare"]["url"], output_file)


def download_wikitext2(output_dir: Path) -> bool:
    """Download WikiText-2 dataset."""
    zip_path = output_dir / "wikitext-2-raw.zip"
    if download_file(DATASETS["wikitext-2"]["url"], zip_path):
        return extract_text_from_wikitext(zip_path, output_dir) is not None
    return False


def download_enwik8(output_dir: Path) -> bool:
    """Download enwik8 dataset."""
    zip_path = output_dir / "enwik8.zip"
    if download_file(DATASETS["enwik8"]["url"], zip_path):
        extract_archive(zip_path, output_dir)
        # enwik8 is a single file
        for f in output_dir.iterdir():
            if f.name.startswith("enwik"):
                return True
    return False


def download_custom(output_dir: Path, url: str, filename: Optional[str] = None) -> bool:
    """Download from custom URL."""
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    
    output_file = output_dir / filename
    return download_file(url, output_file)


def list_datasets():
    """List available datasets."""
    print("\n" + "=" * 60)
    print("Available Datasets")
    print("=" * 60)
    
    for name, info in DATASETS.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
        if 'note' in info:
            print(f"  Note: {info['note']}")
    
    print("\n" + "=" * 60)
    print("\nUsage:")
    print("  python scripts/download_data.py --dataset <name> --output <dir>")
    print("\nExamples:")
    print("  python scripts/download_data.py --dataset shakespeare --output data/raw/shakespeare")
    print("  python scripts/download_data.py --dataset wikitext-2 --output data/raw/wikitext2")
    print("  python scripts/download_data.py --dataset enwik8 --output data/raw/enwik8")
    print("  python scripts/download_data.py --url <custom_url> --output data/raw/custom")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download datasets for DiffusionLM training")
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset name to download (see --list for options)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="Custom URL to download from"
    )
    parser.add_argument(
        "--filename", "-f",
        type=str,
        help="Filename for custom download"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        list_datasets()
        return
    
    # Validate arguments
    if not args.dataset and not args.url:
        print("Error: Must specify --dataset or --url")
        print("Use --list to see available datasets")
        sys.exit(1)
    
    if not args.output:
        print("Error: Must specify --output directory")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    success = False
    
    if args.url:
        print(f"Downloading from custom URL: {args.url}")
        success = download_custom(output_dir, args.url, args.filename)
    
    elif args.dataset:
        dataset_name = args.dataset.lower()
        
        if dataset_name not in DATASETS:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print("Use --list to see available datasets")
            sys.exit(1)
        
        dataset_info = DATASETS[dataset_name]
        print(f"\nDownloading: {dataset_name}")
        print(f"  Description: {dataset_info['description']}")
        print(f"  Size: {dataset_info['size']}")
        print(f"  Output: {output_dir}")
        print()
        
        if dataset_name in ["shakespeare", "tinyshakespeare"]:
            success = download_shakespeare(output_dir)
        elif dataset_name == "wikitext-2":
            success = download_wikitext2(output_dir)
        elif dataset_name == "wikitext-103":
            zip_path = output_dir / DATASETS[dataset_name]["filename"]
            if download_file(dataset_info["url"], zip_path):
                success = extract_archive(zip_path, output_dir)
        elif dataset_name == "enwik8":
            success = download_enwik8(output_dir)
        elif dataset_name == "enwik9":
            zip_path = output_dir / DATASETS[dataset_name]["filename"]
            if download_file(dataset_info["url"], zip_path):
                success = extract_archive(zip_path, output_dir)
        else:
            print(f"Dataset '{dataset_name}' not yet implemented")
            print("Use --url to download from a custom URL")
            sys.exit(1)
    
    if success:
        print(f"\n{'='*60}")
        print("Download complete!")
        print(f"{'='*60}")
        print(f"\nData saved to: {output_dir}")
        print("\nNext steps:")
        print(f"  1. Prepare the data:")
        print(f"     python scripts/prepare_data.py --input {output_dir} --output data/processed")
        print(f"\n  2. Train a model:")
        print(f"     python scripts/train.py --data-dir data/processed --model-preset base")
    else:
        print("\nDownload failed!")
        sys.exit(1)


if __name__ == "__main__":
    if not HAS_REQUESTS:
        print("Error: 'requests' library required.")
        print("Install with: pip install requests")
        print("Or run: ./setup.sh --all")
        sys.exit(1)
    
    main()
