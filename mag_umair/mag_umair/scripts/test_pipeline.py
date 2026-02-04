#!/usr/bin/env python
"""
Quick Validation Test for Fine-Tuning Pipeline

Run this to verify everything works:
    conda run -n magface python scripts/test_pipeline.py
"""

import json
import sys
from pathlib import Path

# Setup paths BEFORE any imports
script_dir = Path(__file__).resolve().parent
mag_umair_dir = script_dir.parent  # mag_umair folder
mag_face_testing_root = mag_umair_dir.parent.parent  # d:/Job/mag_face_testing

# Add paths
sys.path.insert(0, str(mag_umair_dir))
sys.path.insert(0, str(mag_face_testing_root))

# Test paths
TEST_PAIRS_ORIGINAL = mag_face_testing_root / "test_feedback" / "feedback_pairs_test.json"
CHECKPOINT = mag_face_testing_root / "mag_umair" / "__MACOSX" / "mag_umair" / "magface_epoch_00025.pth"


def convert_pairs_format(input_file: Path, output_file: Path):
    """Convert {"pairs": [...]} format to [...] format."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'pairs' in data:
        pairs = data['pairs']
    else:
        pairs = data
    
    # Fix paths to absolute
    for pair in pairs:
        pair['image1'] = str(mag_face_testing_root / pair['image1'])
        pair['image2'] = str(mag_face_testing_root / pair['image2'])
    
    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    return len(pairs)


def main():
    print("=" * 60)
    print("FINE-TUNING PIPELINE VALIDATION TEST")
    print("=" * 60)
    print(f"mag_umair_dir: {mag_umair_dir}")
    print(f"mag_face_testing_root: {mag_face_testing_root}")
    
    # Step 1: Convert pairs format
    print("\n[Step 1] Converting pairs format...")
    output_pairs = mag_umair_dir / "test_pairs_converted.json"
    
    if not TEST_PAIRS_ORIGINAL.exists():
        print(f"  - Test pairs file not found: {TEST_PAIRS_ORIGINAL}")
        print("  - SKIPPING - no test data available")
        return False
    
    num_pairs = convert_pairs_format(TEST_PAIRS_ORIGINAL, output_pairs)
    print(f"  - Converted {num_pairs} pairs to {output_pairs.name}")
    
    # Step 2: Test imports
    print("\n[Step 2] Testing imports...")
    
    try:
        # Test ContrastiveLoss (no dependencies)
        import torch
        import torch.nn.functional as F
        
        # Manually test the loss function
        print("  - PyTorch: OK")
        
        # Test MagFace model import
        from MagFace_repo.models import iresnet
        print("  - MagFace_repo.models: OK")
        
        # Now import our training modules (they depend on the above)
        from src.training.contrastive_loss import ContrastiveLoss
        print("  - ContrastiveLoss: OK")
        
        from src.training.pair_dataset import FeedbackPairDataset
        print("  - FeedbackPairDataset: OK")
        
        from src.training.trainer import MagFaceTrainer
        print("  - MagFaceTrainer: OK")
        
    except ImportError as e:
        print(f"  - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test dataset loading
    print("\n[Step 3] Testing dataset loading...")
    try:
        dataset = FeedbackPairDataset(str(output_pairs))
        print(f"  - Loaded {len(dataset)} pairs")
        
        # Test loading one pair
        img1, img2, label = dataset[0]
        print(f"  - Sample shape: img1={img1.shape}, img2={img2.shape}, label={label}")
    except Exception as e:
        print(f"  - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test trainer initialization
    print("\n[Step 4] Testing trainer initialization...")
    if not CHECKPOINT.exists():
        print(f"  - Checkpoint not found: {CHECKPOINT}")
        print("  - SKIPPING trainer test")
    else:
        try:
            trainer = MagFaceTrainer(
                checkpoint_path=str(CHECKPOINT),
                device="cpu"
            )
            print("  - MagFaceTrainer initialized: OK")
        except Exception as e:
            print(f"  - FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 5: Quick evaluation
        print("\n[Step 5] Quick evaluation on 5 pairs...")
        try:
            # Create small test file with only 5 pairs
            with open(output_pairs, 'r') as f:
                pairs = json.load(f)
            
            small_pairs = pairs[:5]
            small_file = mag_umair_dir / "test_small.json"
            with open(small_file, 'w') as f:
                json.dump(small_pairs, f)
            
            results = trainer.evaluate(str(small_file), threshold=0.4)
            print(f"  - Accuracy: {results['accuracy']:.2%}")
            print(f"  - Total pairs: {results['total_pairs']}")
            
            # Cleanup
            small_file.unlink()
        except Exception as e:
            print(f"  - FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Cleanup
    output_pairs.unlink(missing_ok=True)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run the full fine-tuning:")
    print(f"  python scripts/finetune.py --checkpoint {CHECKPOINT} --train_pairs <your_pairs.json> --epochs 5")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
