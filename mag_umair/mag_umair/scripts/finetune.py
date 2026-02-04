#!/usr/bin/env python
"""
Fine-tune MagFace on Feedback Pairs

Usage:
    python scripts/finetune.py \
        --checkpoint models/magface_epoch_00025.pth \
        --train_pairs path/to/train_pairs.json \
        --val_pairs path/to/val_pairs.json \
        --epochs 10 \
        --output_dir checkpoints/

This will fine-tune the pretrained MagFace model on your labeled pairs
and save checkpoints to the output directory.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import MagFaceTrainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MagFace on feedback pairs")
    
    # Required arguments
    parser.add_argument("--checkpoint", required=True,
                        help="Path to pretrained MagFace checkpoint")
    parser.add_argument("--train_pairs", required=True,
                        help="Path to training pairs JSON file")
    
    # Optional arguments
    parser.add_argument("--val_pairs", default=None,
                        help="Path to validation pairs JSON file")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Contrastive loss margin (default: 1.0)")
    parser.add_argument("--output_dir", default="checkpoints",
                        help="Output directory for checkpoints (default: checkpoints)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to use (default: cpu)")
    parser.add_argument("--arch", default="iresnet100",
                        choices=["iresnet100", "iresnet50", "iresnet18"],
                        help="Model architecture (default: iresnet100)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MagFaceTrainer(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        device=args.device
    )
    
    # Fine-tune
    history = trainer.finetune(
        train_pairs_file=args.train_pairs,
        val_pairs_file=args.val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        margin=args.margin,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_accuracy']:
        print(f"Best validation accuracy: {history['best_accuracy']:.2%} (epoch {history['best_epoch']})")
    print(f"Checkpoints saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
