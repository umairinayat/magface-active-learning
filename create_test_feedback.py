#!/usr/bin/env python
"""
Create Test Feedback Dataset

This script:
1. Extracts 50 images from CASIA-WebFace dataset (10 identities × 5 images each)
2. Generates synthetic feedback pairs with both correct and incorrect labels
3. Simulates real-world scenario where users make some mistakes
4. Creates test dataset for validating fine-tuning pipeline
"""
import sys
sys.path.insert(0, 'MagFace_repo')

import os
import json
import random
import shutil
import mxnet as mx
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class TestFeedbackGenerator:
    """
    Generate test feedback dataset from CASIA-WebFace.
    """
    
    def __init__(self, dataset_path, output_dir='test_feedback'):
        """
        Args:
            dataset_path: Path to CASIA-WebFace MXNet dataset
            output_dir: Output directory for test images and feedback
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        
        print(f"✅ Output directory: {output_dir}")
    
    def load_dataset(self):
        """Load CASIA-WebFace dataset."""
        print(f"\nLoading dataset from {self.dataset_path}...")
        
        # Find .rec and .idx files
        rec_path = os.path.join(self.dataset_path, 'train.rec')
        idx_path = os.path.join(self.dataset_path, 'train.idx')
        
        if not os.path.exists(rec_path):
            raise FileNotFoundError(f"Dataset not found: {rec_path}")
        
        # Load MXNet dataset
        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        
        print(f"✅ Dataset loaded")
        
        return imgrec
    
    def extract_images(self, imgrec, num_identities=10, images_per_identity=5):
        """
        Extract images from dataset.
        
        Args:
            imgrec: MXNet record
            num_identities: Number of different people
            images_per_identity: Images per person
        
        Returns:
            identity_to_images: Dict mapping identity_id to list of image paths
        """
        print(f"\nExtracting {num_identities} identities × {images_per_identity} images...")
        
        # Scan dataset to find identities
        identity_images = defaultdict(list)
        
        print("Scanning dataset...")
        idx = 0
        pbar = tqdm(total=num_identities * images_per_identity * 2)  # Estimate
        
        while len(identity_images) < num_identities or \
              any(len(imgs) < images_per_identity for imgs in identity_images.values()):
            
            try:
                s = imgrec.read_idx(idx)
                if s is None:
                    break
                
                header, img = mx.recordio.unpack(s)
                label = int(header.label)
                
                # Only collect if we need more from this identity
                if label not in identity_images or len(identity_images[label]) < images_per_identity:
                    # Decode image
                    img_array = mx.image.imdecode(img).asnumpy()
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Save image
                    img_filename = f"identity_{label:04d}_img_{len(identity_images[label]):02d}.jpg"
                    img_path = os.path.join(self.images_dir, img_filename)
                    cv2.imwrite(img_path, img_bgr)
                    
                    identity_images[label].append(img_path)
                    pbar.update(1)
                
                # Check if we have enough
                complete_identities = [
                    identity for identity, imgs in identity_images.items()
                    if len(imgs) >= images_per_identity
                ]
                
                if len(complete_identities) >= num_identities:
                    break
                
                idx += 1
                
            except Exception as e:
                idx += 1
                continue
        
        pbar.close()
        
        # Keep only complete identities
        identity_to_images = {
            identity: imgs[:images_per_identity]
            for identity, imgs in identity_images.items()
            if len(imgs) >= images_per_identity
        }
        
        # Keep only requested number
        identity_to_images = dict(list(identity_to_images.items())[:num_identities])
        
        total_images = sum(len(imgs) for imgs in identity_to_images.values())
        print(f"✅ Extracted {total_images} images from {len(identity_to_images)} identities")
        
        return identity_to_images
    
    def generate_feedback_pairs(self, identity_to_images, 
                                num_positive=100, num_negative=100,
                                correct_rate=0.8):
        """
        Generate synthetic feedback pairs.
        
        Args:
            identity_to_images: Dict mapping identity to image paths
            num_positive: Number of positive pairs (same person)
            num_negative: Number of negative pairs (different person)
            correct_rate: Ratio of correct labels (0.8 = 80% correct, 20% wrong)
        
        Returns:
            pairs: List of feedback pairs
            ground_truth: List of actual ground truth labels
        """
        print(f"\nGenerating feedback pairs...")
        print(f"  Positive pairs (same person): {num_positive}")
        print(f"  Negative pairs (different person): {num_negative}")
        print(f"  Correct label rate: {correct_rate*100:.0f}%")
        
        pairs = []
        ground_truth = []
        
        identities = list(identity_to_images.keys())
        
        # Generate positive pairs (same person)
        print("\nGenerating positive pairs...")
        for _ in tqdm(range(num_positive)):
            # Pick random identity
            identity = random.choice(identities)
            images = identity_to_images[identity]
            
            # Pick two different images from same person
            if len(images) >= 2:
                img1, img2 = random.sample(images, 2)
                
                # Ground truth: same person
                actual_label = 1
                
                # Simulate user error
                if random.random() < correct_rate:
                    # Correct label
                    user_label = 1
                else:
                    # User makes mistake (says different when actually same)
                    user_label = 0
                
                pairs.append({
                    'image1': img1,
                    'image2': img2,
                    'label': user_label,
                    'identity1': identity,
                    'identity2': identity
                })
                
                ground_truth.append({
                    'image1': img1,
                    'image2': img2,
                    'actual_label': actual_label,
                    'user_label': user_label,
                    'correct': (actual_label == user_label)
                })
        
        # Generate negative pairs (different person)
        print("Generating negative pairs...")
        for _ in tqdm(range(num_negative)):
            # Pick two different identities
            if len(identities) >= 2:
                identity1, identity2 = random.sample(identities, 2)
                
                # Pick one image from each
                img1 = random.choice(identity_to_images[identity1])
                img2 = random.choice(identity_to_images[identity2])
                
                # Ground truth: different person
                actual_label = 0
                
                # Simulate user error
                if random.random() < correct_rate:
                    # Correct label
                    user_label = 0
                else:
                    # User makes mistake (says same when actually different)
                    user_label = 1
                
                pairs.append({
                    'image1': img1,
                    'image2': img2,
                    'label': user_label,
                    'identity1': identity1,
                    'identity2': identity2
                })
                
                ground_truth.append({
                    'image1': img1,
                    'image2': img2,
                    'actual_label': actual_label,
                    'user_label': user_label,
                    'correct': (actual_label == user_label)
                })
        
        # Shuffle pairs
        combined = list(zip(pairs, ground_truth))
        random.shuffle(combined)
        pairs, ground_truth = zip(*combined)
        pairs = list(pairs)
        ground_truth = list(ground_truth)
        
        # Statistics
        correct_count = sum(1 for gt in ground_truth if gt['correct'])
        incorrect_count = len(ground_truth) - correct_count
        
        print(f"\n✅ Generated {len(pairs)} feedback pairs")
        print(f"   Correct labels: {correct_count} ({correct_count/len(pairs)*100:.1f}%)")
        print(f"   Incorrect labels: {incorrect_count} ({incorrect_count/len(pairs)*100:.1f}%)")
        
        return pairs, ground_truth
    
    def save_feedback(self, pairs, ground_truth):
        """Save feedback pairs and ground truth."""
        # Save user feedback (what model will see)
        feedback_file = os.path.join(self.output_dir, 'feedback_pairs_test.json')
        with open(feedback_file, 'w') as f:
            json.dump({'pairs': pairs}, f, indent=2)
        print(f"\n✅ Saved feedback pairs: {feedback_file}")
        
        # Save ground truth (for evaluation)
        ground_truth_file = os.path.join(self.output_dir, 'ground_truth.json')
        with open(ground_truth_file, 'w') as f:
            json.dump({'pairs': ground_truth}, f, indent=2)
        print(f"✅ Saved ground truth: {ground_truth_file}")
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'dataset_info.txt')
        with open(stats_file, 'w') as f:
            f.write("Test Feedback Dataset Statistics\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total pairs: {len(pairs)}\n")
            f.write(f"Positive pairs (same person): {sum(1 for p in pairs if p['label'] == 1)}\n")
            f.write(f"Negative pairs (different person): {sum(1 for p in pairs if p['label'] == 0)}\n\n")
            
            correct_count = sum(1 for gt in ground_truth if gt['correct'])
            f.write(f"Correct labels: {correct_count} ({correct_count/len(pairs)*100:.1f}%)\n")
            f.write(f"Incorrect labels: {len(pairs) - correct_count} ({(len(pairs) - correct_count)/len(pairs)*100:.1f}%)\n\n")
            
            # Break down by type
            correct_positive = sum(1 for gt in ground_truth if gt['actual_label'] == 1 and gt['correct'])
            incorrect_positive = sum(1 for gt in ground_truth if gt['actual_label'] == 1 and not gt['correct'])
            correct_negative = sum(1 for gt in ground_truth if gt['actual_label'] == 0 and gt['correct'])
            incorrect_negative = sum(1 for gt in ground_truth if gt['actual_label'] == 0 and not gt['correct'])
            
            f.write("Breakdown:\n")
            f.write(f"  Same person pairs:\n")
            f.write(f"    Correctly labeled as 'same': {correct_positive}\n")
            f.write(f"    Incorrectly labeled as 'different': {incorrect_positive}\n")
            f.write(f"  Different person pairs:\n")
            f.write(f"    Correctly labeled as 'different': {correct_negative}\n")
            f.write(f"    Incorrectly labeled as 'same': {incorrect_negative}\n")
        
        print(f"✅ Saved statistics: {stats_file}")
        
        return feedback_file, ground_truth_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test feedback dataset')
    parser.add_argument('--dataset', type=str, 
                        default='faces_webface_112x112',
                        help='Path to CASIA-WebFace dataset')
    parser.add_argument('--output_dir', type=str, default='test_feedback',
                        help='Output directory')
    parser.add_argument('--num_identities', type=int, default=10,
                        help='Number of identities to extract')
    parser.add_argument('--images_per_identity', type=int, default=5,
                        help='Images per identity')
    parser.add_argument('--num_positive', type=int, default=100,
                        help='Number of positive pairs')
    parser.add_argument('--num_negative', type=int, default=100,
                        help='Number of negative pairs')
    parser.add_argument('--correct_rate', type=float, default=1.0,
                        help='Ratio of correct labels (1.0 = 100%% correct, no noise)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Creating Test Feedback Dataset")
    print("="*70)
    
    # Initialize generator
    generator = TestFeedbackGenerator(args.dataset, args.output_dir)
    
    # Load dataset
    imgrec = generator.load_dataset()
    
    # Extract images
    identity_to_images = generator.extract_images(
        imgrec,
        num_identities=args.num_identities,
        images_per_identity=args.images_per_identity
    )
    
    # Generate feedback pairs
    pairs, ground_truth = generator.generate_feedback_pairs(
        identity_to_images,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        correct_rate=args.correct_rate
    )
    
    # Save
    feedback_file, ground_truth_file = generator.save_feedback(pairs, ground_truth)
    
    print("\n" + "="*70)
    print("✅ Test Dataset Created Successfully!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Fine-tune on synthetic feedback:")
    print(f"   python finetune_on_feedback.py \\")
    print(f"     --checkpoint checkpoints/magface_finetuned_final.pth \\")
    print(f"     --feedback_file {feedback_file} \\")
    print(f"     --epochs 10 \\")
    print(f"     --force_cpu")
    print(f"\n2. Evaluate results:")
    print(f"   python evaluate_finetuned_model.py \\")
    print(f"     --pretrained checkpoints/magface_finetuned_final.pth \\")
    print(f"     --finetuned checkpoints_feedback/magface_feedback_best.pth \\")
    print(f"     --feedback_file {feedback_file} \\")
    print(f"     --force_cpu")
    print(f"\n3. Check ground truth accuracy:")
    print(f"   Compare with {ground_truth_file}")


if __name__ == '__main__':
    main()
