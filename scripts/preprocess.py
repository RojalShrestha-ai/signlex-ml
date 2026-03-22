#!/usr/bin/env python3
import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


# Label mapping: A-Z → 0-25
ASL_LABELS = {chr(i): i - ord('A') for i in range(ord('A'), ord('Z') + 1)}
ASL_LABELS_INV = {v: k for k, v in ASL_LABELS.items()}

# Additional labels some datasets use
EXTRA_LABEL_MAP = {
    'del': None,      # Skip - not a letter
    'delete': None,   # Skip
    'nothing': None,  # Skip
    'space': None,    # Skip
    '0': None, '1': None, '2': None, '3': None, '4': None,  # Skip numbers
    '5': None, '6': None, '7': None, '8': None, '9': None,
}


def normalize_label(label: str) -> Optional[int]:

    label = str(label).strip().upper()
    
    # Check if it's a letter A-Z
    if label in ASL_LABELS:
        return ASL_LABELS[label]
    
    # Check extra mappings
    if label.lower() in EXTRA_LABEL_MAP:
        return EXTRA_LABEL_MAP[label.lower()]
    
    # Try numeric (Sign MNIST style: 0-8 = A-I, 10-24 = K-Y)
    try:
        num = int(label)
        if 0 <= num <= 8:
            return num  # A-I
        elif 10 <= num <= 24:
            return num  # K-Y (J=9 is skipped)
        else:
            return None
    except ValueError:
        pass
    
    return None


def process_asl_alphabet(data_dir: Path) -> List[Dict]:

    samples = []
    dataset_path = data_dir / "asl_alphabet" / "asl_alphabet_train"
    
    if not dataset_path.exists():
        # Try alternate path
        dataset_path = data_dir / "asl_alphabet"
    
    if not dataset_path.exists():
        print(f"Warning: ASL Alphabet dataset not found at {dataset_path}")
        return samples
    
    print(f"Processing ASL Alphabet from {dataset_path}...")
    
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        label = normalize_label(class_dir.name)
        if label is None:
            print(f"  Skipping class: {class_dir.name}")
            continue
        
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in images:
            samples.append({
                "path": str(img_path.absolute()),
                "label": label,
                "source": "asl_alphabet"
            })
    
    print(f"  Found {len(samples)} samples")
    return samples


def process_asl_hg_raw(data_dir: Path) -> List[Dict]:

    samples = []
    dataset_path = data_dir / "asl_hg_raw" / "asl_dataset"
    
    if not dataset_path.exists():
        print(f"Warning: ASL HG Raw dataset not found at {dataset_path}")
        return samples
    
    print(f"Processing ASL HG Raw from {dataset_path}...")
    
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        label = normalize_label(class_dir.name)
        if label is None:
            continue
        
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        for img_path in images:
            samples.append({
                "path": str(img_path.absolute()),
                "label": label,
                "source": "asl_hg_raw"
            })
    
    print(f"  Found {len(samples)} samples")
    return samples


def process_asl_hg_processed(data_dir: Path) -> List[Dict]:

    samples = []
    dataset_path = data_dir / "asl_hg_processed" / "asl_processed" / "train"
    
    if not dataset_path.exists():
        print(f"Warning: ASL HG Processed dataset not found at {dataset_path}")
        return samples
    
    print(f"Processing ASL HG Processed from {dataset_path}...")
    
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        label = normalize_label(class_dir.name)
        if label is None:
            continue
        
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        for img_path in images:
            samples.append({
                "path": str(img_path.absolute()),
                "label": label,
                "source": "asl_hg_processed"
            })
    
    print(f"  Found {len(samples)} samples")
    return samples


def process_mendeley_210k(data_dir: Path, include_types: List[str] = None) -> List[Dict]:

    samples = []
    
    # Default: use ALL gesture types for maximum data
    if include_types is None:
        include_types = [
            "Type_01_(Raw_Gesture)",
            "Type_02_(Keypoint Based)", 
            "Type_03_(Skeleton_Overlay)",
            "Type_04_(Isolated_Skeleton)",
            "Type_05_(Enhanced_Skeleton)"
        ]
    
    for type_name in include_types:
        dataset_path = data_dir / "mendeley_210k" / "Root" / type_name
        
        if not dataset_path.exists():
            print(f"Warning: Mendeley {type_name} not found at {dataset_path}")
            continue
        
        print(f"Processing Mendeley {type_name}...")
        
        # Find the alphabet train folder
        train_path = dataset_path / "asl_alphabet_train"
        if not train_path.exists():
            train_path = dataset_path
        
        for class_dir in sorted(train_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            label = normalize_label(class_dir.name)
            if label is None:
                continue
            
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            for img_path in images:
                samples.append({
                    "path": str(img_path.absolute()),
                    "label": label,
                    "source": f"mendeley_{type_name}"
                })
    
    print(f"  Found {len(samples)} samples total from Mendeley")
    return samples


def process_sign_mnist(data_dir: Path, output_dir: Path) -> List[Dict]:

    import pandas as pd
    
    samples = []
    
    # Find CSV files
    train_csv = data_dir / "sign_mnist" / "sign_mnist_train.csv"
    test_csv = data_dir / "sign_mnist" / "sign_mnist_test.csv"
    
    csv_files = []
    if train_csv.exists():
        csv_files.append(train_csv)
    if test_csv.exists():
        csv_files.append(test_csv)
    
    if not csv_files:
        print(f"Warning: Sign MNIST CSVs not found in {data_dir / 'sign_mnist'}")
        return samples
    
    # Create output directory for converted images
    mnist_output = output_dir / "sign_mnist_images"
    mnist_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing Sign MNIST...")
    
    for csv_path in csv_files:
        print(f"  Reading {csv_path.name}...")
        df = pd.read_csv(csv_path)
        
        labels = df.iloc[:, 0].values
        pixels = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        
        for i, (label, pixel_data) in enumerate(tqdm(zip(labels, pixels), total=len(labels), desc="  Converting")):
            # Normalize label (Sign MNIST uses 0-24, skipping J=9)
            norm_label = normalize_label(str(label))
            if norm_label is None:
                continue
            
            # Save as PNG
            img_name = f"{csv_path.stem}_{i:06d}.png"
            img_path = mnist_output / img_name
            
            img = Image.fromarray(pixel_data, mode='L')
            img = img.convert('RGB')  # Convert to RGB
            img.save(img_path)
            
            samples.append({
                "path": str(img_path.absolute()),
                "label": norm_label,
                "source": "sign_mnist"
            })
    
    print(f"  Found {len(samples)} samples")
    return samples


def process_signalphaset(data_dir: Path) -> List[Dict]:

    samples = []
    dataset_path = data_dir / "signalphaset_static" / "SignAlphaSet"
    
    if not dataset_path.exists():
        print(f"Warning: SignAlphaSet not found at {dataset_path}")
        return samples
    
    print(f"Processing SignAlphaSet from {dataset_path}...")
    
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        label = normalize_label(class_dir.name)
        if label is None:
            continue
        
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in images:
            samples.append({
                "path": str(img_path.absolute()),
                "label": label,
                "source": "signalphaset"
            })
    
    print(f"  Found {len(samples)} samples")
    return samples


def validate_image(path: str) -> bool:
    """Check if image is valid and readable."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def stratified_split(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    random.seed(seed)
    
    # Group by label
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample["label"]].append(sample)
    
    train, val, test = [], [], []
    
    for label, label_samples in by_label.items():
        random.shuffle(label_samples)
        n = len(label_samples)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train.extend(label_samples[:n_train])
        val.extend(label_samples[n_train:n_train + n_val])
        test.extend(label_samples[n_train + n_val:])
    
    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def print_statistics(samples: List[Dict], name: str):

    by_label = defaultdict(int)
    by_source = defaultdict(int)
    
    for s in samples:
        by_label[s["label"]] += 1
        by_source[s.get("source", "unknown")] += 1
    
    print(f"\n{'='*60}")
    print(f"{name}: {len(samples)} samples")
    print(f"{'='*60}")
    
    print("\nBy class:")
    for label in sorted(by_label.keys()):
        letter = ASL_LABELS_INV[label]
        count = by_label[label]
        print(f"  {letter} ({label}): {count:,}")
    
    print("\nBy source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ASL datasets")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="~/utils/data/raw",
        help="Directory containing raw datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for manifests"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--validate_images",
        action="store_true",
        help="Validate each image (slower but safer)"
    )
    parser.add_argument(
        "--include_skeleton",
        action="store_true",
        help="Include skeleton-based Mendeley types"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ASL Dataset Preprocessing")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Collect all samples
    all_samples = []
    
    # Process each dataset
    all_samples.extend(process_asl_alphabet(data_dir))
    all_samples.extend(process_asl_hg_raw(data_dir))
    all_samples.extend(process_asl_hg_processed(data_dir))
    
    # Mendeley - choose which types
    mendeley_types = None
    if args.include_skeleton:
        mendeley_types.extend([
            "Type_02_(Keypoint Based)",
            "Type_03_(Skeleton_Overlay)",
        ])
    all_samples.extend(process_mendeley_210k(data_dir, mendeley_types))
    
    # Sign MNIST (requires conversion)
    try:
        import pandas as pd
        all_samples.extend(process_sign_mnist(data_dir, output_dir))
    except ImportError:
        print("Warning: pandas not installed, skipping Sign MNIST")
    
    all_samples.extend(process_signalphaset(data_dir))
    
    print(f"\n{'='*60}")
    print(f"Total samples collected: {len(all_samples)}")
    print(f"{'='*60}")
    
    # Optional: validate images
    if args.validate_images:
        print("\nValidating images...")
        valid_samples = []
        for sample in tqdm(all_samples, desc="Validating"):
            if validate_image(sample["path"]):
                valid_samples.append(sample)
        
        print(f"Valid: {len(valid_samples)} / {len(all_samples)}")
    
    # Split into train/val/test
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    train, val, test = stratified_split(
        all_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        seed=args.seed
    )
    
    # Print statistics
    print_statistics(train, "TRAIN SET")
    print_statistics(val, "VALIDATION SET")
    print_statistics(test, "TEST SET")
    
    # Remove source field for final manifests (not needed during training)
    def clean_manifest(samples):
        return [{"path": s["path"], "label": s["label"]} for s in samples]
    
    # Save manifests
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    test_path = output_dir / "test.json"
    
    with open(train_path, 'w') as f:
        json.dump(clean_manifest(train), f, indent=2)
    print(f"\nSaved: {train_path}")
    
    with open(val_path, 'w') as f:
        json.dump(clean_manifest(val), f, indent=2)
    print(f"Saved: {val_path}")
    
    with open(test_path, 'w') as f:
        json.dump(clean_manifest(test), f, indent=2)
    print(f"Saved: {test_path}")
    
    # Save metadata
    metadata = {
        "num_classes": 26,
        "classes": ASL_LABELS,
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "total_samples": len(all_samples),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": test_ratio,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {output_dir / 'metadata.json'}")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nNext step: Run training")
    print(f"  python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()