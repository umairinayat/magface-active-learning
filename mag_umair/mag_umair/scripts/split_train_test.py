#!/usr/bin/env python
"""
Split feedback pairs into train/test sets

Usage:
    python scripts/split_train_test.py \
        --input path/to/feedback_pairs.json \
        --train_output train_pairs.json \
        --test_output test_pairs.json \
        --split 0.8

This creates separate train and test files from your feedback data.
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split pairs into train/test")
    
    parser.add_argument("--input", required=True,
                        help="Input pairs JSON file")
    parser.add_argument("--train_output", default="train_pairs.json",
                        help="Output file for training pairs")
    parser.add_argument("--test_output", default="test_pairs.json",
                        help="Output file for test pairs")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train split ratio (default: 0.8 = 80% train, 20% test)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load input pairs
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Handle both formats
    if isinstance(data, dict) and 'pairs' in data:
        pairs = data['pairs']
    else:
        pairs = data
    
    print(f"Loaded {len(pairs)} pairs from {args.input}")
    
    # Shuffle
    random.seed(args.seed)
    random.shuffle(pairs)
    
    # Split
    split_idx = int(len(pairs) * args.split)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    # Count positive/negative in each
    train_pos = sum(1 for p in train_pairs if p['label'] == 1)
    train_neg = len(train_pairs) - train_pos
    test_pos = sum(1 for p in test_pairs if p['label'] == 1)
    test_neg = len(test_pairs) - test_pos
    
    print(f"\nTrain set: {len(train_pairs)} pairs")
    print(f"  Positive (same): {train_pos}")
    print(f"  Negative (different): {train_neg}")
    
    print(f"\nTest set: {len(test_pairs)} pairs")
    print(f"  Positive (same): {test_pos}")
    print(f"  Negative (different): {test_neg}")
    
    # Save
    with open(args.train_output, 'w') as f:
        json.dump(train_pairs, f, indent=2)
    print(f"\nTrain pairs saved to: {args.train_output}")
    
    with open(args.test_output, 'w') as f:
        json.dump(test_pairs, f, indent=2)
    print(f"Test pairs saved to: {args.test_output}")


if __name__ == "__main__":
    main()
