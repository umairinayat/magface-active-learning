#!/usr/bin/env python
"""
Validate Fine-Tuning - Compare Pretrained vs Fine-tuned Model

Usage:
    python scripts/validate_finetuning.py \
        --pretrained models/magface_epoch_00025.pth \
        --finetuned checkpoints/magface_finetuned_best.pth \
        --pairs path/to/test_pairs.json

This script generates a validation report comparing both models,
which you can use to prove to your mentor that fine-tuning works.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import MagFaceTrainer


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuning by comparing models")
    
    parser.add_argument("--pretrained", required=True,
                        help="Path to pretrained MagFace checkpoint")
    parser.add_argument("--finetuned", required=True,
                        help="Path to fine-tuned MagFace checkpoint")
    parser.add_argument("--pairs", required=True,
                        help="Path to test pairs JSON file")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Similarity threshold (default: 0.4)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to use (default: cpu)")
    parser.add_argument("--output", default="validation_report.json",
                        help="Output report file (default: validation_report.json)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FINE-TUNING VALIDATION REPORT")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pretrained model: {args.pretrained}")
    print(f"Fine-tuned model: {args.finetuned}")
    print(f"Test pairs: {args.pairs}")
    print(f"Threshold: {args.threshold}")
    print("=" * 70)
    
    # Evaluate pretrained model
    print("\n[1/2] Evaluating PRETRAINED model...")
    pretrained_trainer = MagFaceTrainer(
        checkpoint_path=args.pretrained,
        device=args.device
    )
    pretrained_results = pretrained_trainer.evaluate(args.pairs, threshold=args.threshold)
    
    # Evaluate fine-tuned model
    print("\n[2/2] Evaluating FINE-TUNED model...")
    finetuned_trainer = MagFaceTrainer(
        checkpoint_path=args.finetuned,
        device=args.device
    )
    finetuned_results = finetuned_trainer.evaluate(args.pairs, threshold=args.threshold)
    
    # Calculate improvements
    acc_improvement = finetuned_results['accuracy'] - pretrained_results['accuracy']
    fp_improvement = pretrained_results['false_positives'] - finetuned_results['false_positives']
    fn_improvement = pretrained_results['false_negatives'] - finetuned_results['false_negatives']
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON: PRETRAINED vs FINE-TUNED")
    print("=" * 70)
    print(f"{'Metric':<25} {'Pretrained':>15} {'Fine-tuned':>15} {'Change':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {pretrained_results['accuracy']:>14.2%} {finetuned_results['accuracy']:>14.2%} {acc_improvement:>+14.2%}")
    print(f"{'Precision':<25} {pretrained_results['precision']:>14.2%} {finetuned_results['precision']:>14.2%}")
    print(f"{'Recall':<25} {pretrained_results['recall']:>14.2%} {finetuned_results['recall']:>14.2%}")
    print(f"{'F1 Score':<25} {pretrained_results['f1']:>14.2%} {finetuned_results['f1']:>14.2%}")
    print("-" * 70)
    print(f"{'True Positives':<25} {pretrained_results['true_positives']:>15} {finetuned_results['true_positives']:>15}")
    print(f"{'True Negatives':<25} {pretrained_results['true_negatives']:>15} {finetuned_results['true_negatives']:>15}")
    print(f"{'False Positives':<25} {pretrained_results['false_positives']:>15} {finetuned_results['false_positives']:>15} {fp_improvement:>+15}")
    print(f"{'False Negatives':<25} {pretrained_results['false_negatives']:>15} {finetuned_results['false_negatives']:>15} {fn_improvement:>+15}")
    print("=" * 70)
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if acc_improvement > 0:
        print(f"✅ FINE-TUNING SUCCESSFUL: Accuracy improved by {acc_improvement:.2%}")
    elif acc_improvement == 0:
        print(f"⚠️  NO CHANGE: Accuracy remained the same")
    else:
        print(f"❌ FINE-TUNING DECREASED ACCURACY by {abs(acc_improvement):.2%}")
    
    if fp_improvement > 0:
        print(f"✅ False positives reduced by {fp_improvement}")
    if fn_improvement > 0:
        print(f"✅ False negatives reduced by {fn_improvement}")
    print("=" * 70)
    
    # Save report (convert numpy types for JSON)
    report = {
        'date': datetime.now().isoformat(),
        'pretrained_model': args.pretrained,
        'finetuned_model': args.finetuned,
        'test_pairs': args.pairs,
        'threshold': args.threshold,
        'pretrained_results': pretrained_results,
        'finetuned_results': finetuned_results,
        'improvements': {
            'accuracy': float(acc_improvement),
            'false_positives_reduced': int(fp_improvement),
            'false_negatives_reduced': int(fn_improvement)
        },
        'success': bool(acc_improvement > 0)
    }
    
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to: {args.output}")
    
    return report


if __name__ == "__main__":
    main()
