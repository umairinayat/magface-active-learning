#!/usr/bin/env python
"""
Evaluate MagFace Model on Pairs

Usage:
    python scripts/evaluate.py \
        --model models/magface_epoch_00025.pth \
        --pairs path/to/test_pairs.json \
        --threshold 0.4

This script evaluates a MagFace model on labeled pairs and outputs metrics.
Use this to compare pretrained vs fine-tuned model performance.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import MagFaceTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate MagFace on pairs")
    
    parser.add_argument("--model", required=True,
                        help="Path to MagFace model checkpoint")
    parser.add_argument("--pairs", required=True,
                        help="Path to test pairs JSON file")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Similarity threshold for classification (default: 0.4)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to use (default: cpu)")
    parser.add_argument("--arch", default="iresnet100",
                        choices=["iresnet100", "iresnet50", "iresnet18"],
                        help="Model architecture (default: iresnet100)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize trainer (just for evaluation)
    trainer = MagFaceTrainer(
        checkpoint_path=args.model,
        arch=args.arch,
        device=args.device
    )
    
    # Evaluate
    print(f"\nEvaluating on {args.pairs}...")
    results = trainer.evaluate(args.pairs, threshold=args.threshold)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total pairs:      {results['total_pairs']}")
    print(f"Threshold:        {results['threshold']}")
    print("-" * 60)
    print(f"Accuracy:         {results['accuracy']:.2%}")
    print(f"Precision:        {results['precision']:.2%}")
    print(f"Recall:           {results['recall']:.2%}")
    print(f"F1 Score:         {results['f1']:.2%}")
    print("-" * 60)
    print(f"True Positives:   {results['true_positives']}")
    print(f"True Negatives:   {results['true_negatives']}")
    print(f"False Positives:  {results['false_positives']}")
    print(f"False Negatives:  {results['false_negatives']}")
    print("=" * 60)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
