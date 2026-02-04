#!/usr/bin/env python
"""
Visualize Model Errors

This script:
1. Loads the fine-tuned model
2. Evaluates on test data
3. Identifies false predictions (FP and FN)
4. Saves error image pairs to visualization folder
"""
import sys
sys.path.insert(0, 'MagFace_repo')

import os
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
import shutil

from MagFace_repo.models.iresnet import iresnet100


class ErrorVisualizer:
    """
    Visualize model errors by saving false prediction image pairs.
    """
    
    def __init__(self, model_path, device='cpu', features_list=None):
        self.device = device
        self.feature_cache = None
        
        print(f"Loading model from {model_path}...")
        self.model = iresnet100(num_classes=512)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract state_dict and clean keys
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Clean keys: features.module.X -> X
        from collections import OrderedDict
        cleaned = OrderedDict()
        model_dict = self.model.state_dict()
        
        for k, v in state_dict.items():
            # Remove prefixes
            if k.startswith('features.module.'):
                new_k = k.replace('features.module.', '')
            elif k.startswith('module.features.'):
                new_k = k.replace('module.features.', '')
            elif k.startswith('module.'):
                new_k = k.replace('module.', '')
            elif k.startswith('features.'):
                new_k = k.replace('features.', '')
            else:
                new_k = k
            
            # Skip classification head (fc.weight with wrong shape)
            if new_k == 'fc.weight' and v.shape[0] != 512:
                continue
            if new_k == 'fc.bias' and v.shape[0] != 512:
                continue
            
            # Only load if shape matches
            if new_k in model_dict and v.shape == model_dict[new_k].shape:
                cleaned[new_k] = v
        
        self.model.load_state_dict(cleaned, strict=False)
        print(f"Loaded {len(cleaned)}/{len(model_dict)} weight tensors")
        
        self.model.to(device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor()
            # MagFace expects [0, 1] range, NO mean/std normalization
        ])
        
        # Optional: load precomputed embeddings to speed up pair evaluation
        if features_list:
            self.feature_cache = self._load_feature_cache(features_list)
            print(f"✅ Loaded feature cache with {len(self.feature_cache)} embeddings")
        else:
            print("✅ Model loaded")

    def _load_feature_cache(self, features_list):
        """Load features.list into memory and L2-normalize embeddings."""
        cache = {}
        with open(features_list, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 513:
                    continue
                path = parts[0]
                emb = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                norm = np.linalg.norm(emb) + 1e-12
                emb = emb / norm
                cache[path] = emb
        return cache
    
    def extract_embedding(self, image_path):
        """Extract normalized embedding from image."""
        if self.feature_cache is not None and image_path in self.feature_cache:
            return self.feature_cache[image_path]

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0]
    
    def compute_cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity."""
        return np.dot(emb1, emb2)
    
    def visualize_errors(self, feedback_file, output_dir='error_visualization', threshold=0.4, use_ground_truth=False, max_fp=None, max_fn=None):
        """
        Identify and save error cases.
        
        Args:
            feedback_file: JSON file with test pairs
            output_dir: Directory to save error visualizations
            threshold: Similarity threshold for classification
            use_ground_truth: If True, use actual_label from ground truth file
        """
        print(f"\n{'='*70}")
        print(f"Visualizing Model Errors")
        print(f"{'='*70}")
        
        # Create output directories
        fp_dir = os.path.join(output_dir, 'false_positives')
        fn_dir = os.path.join(output_dir, 'false_negatives')
        tp_dir = os.path.join(output_dir, 'true_positives')
        tn_dir = os.path.join(output_dir, 'true_negatives')
        
        for dir_path in [fp_dir, fn_dir, tp_dir, tn_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Load pairs
        with open(feedback_file, 'r') as f:
            data = json.load(f)
        pairs = data['pairs']
        
        # Check if using ground truth labels
        if use_ground_truth and 'actual_label' in pairs[0]:
            print(f"Using GROUND TRUTH labels (actual_label)")
            label_key = 'actual_label'
        elif 'actual_label' in pairs[0]:
            print(f"Using GROUND TRUTH labels (actual_label)")
            label_key = 'actual_label'
        else:
            print(f"Using USER labels (label) - may contain noise!")
            label_key = 'label'
        
        print(f"Loaded {len(pairs)} test pairs")
        print(f"Using threshold: {threshold}")
        
        # Evaluate and categorize
        false_positives = []
        false_negatives = []
        true_positives = []
        true_negatives = []
        
        print("\nEvaluating pairs...")
        for idx, pair in enumerate(tqdm(pairs)):
            # Extract embeddings
            emb1 = self.extract_embedding(pair['image1'])
            emb2 = self.extract_embedding(pair['image2'])
            
            # Compute similarity
            similarity = self.compute_cosine_similarity(emb1, emb2)
            
            # Predict
            prediction = 1 if similarity > threshold else 0
            actual = pair.get(label_key, pair.get('label', 0))
            
            # Categorize
            pair_info = {
                'image1': pair['image1'],
                'image2': pair['image2'],
                'similarity': float(similarity),
                'prediction': prediction,
                'actual': actual,
                'idx': idx
            }
            
            if prediction == 1 and actual == 0:
                false_positives.append(pair_info)
            elif prediction == 0 and actual == 1:
                false_negatives.append(pair_info)
            elif prediction == 1 and actual == 1:
                true_positives.append(pair_info)
            elif prediction == 0 and actual == 0:
                true_negatives.append(pair_info)
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"Results Summary")
        print(f"{'='*70}")
        print(f"True Positives (Correct 'Same'):  {len(true_positives)}")
        print(f"True Negatives (Correct 'Diff'):  {len(true_negatives)}")
        print(f"False Positives (Wrong 'Same'):   {len(false_positives)}")
        print(f"False Negatives (Wrong 'Diff'):   {len(false_negatives)}")
        print(f"\nAccuracy: {(len(true_positives) + len(true_negatives)) / len(pairs):.3f}")
        print(f"Precision: {len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0:.3f}")
        print(f"Recall: {len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0:.3f}")
        
        # Save error cases
        print(f"\n{'='*70}")
        print(f"Saving Error Cases")
        print(f"{'='*70}")
        
        self._save_pairs(false_positives, fp_dir, "False Positive", limit=max_fp)
        self._save_pairs(false_negatives, fn_dir, "False Negative", limit=max_fn)
        
        # Save some correct cases for comparison
        print(f"\nSaving sample correct predictions for comparison...")
        self._save_pairs(true_positives[:10], tp_dir, "True Positive", limit=10)
        self._save_pairs(true_negatives[:10], tn_dir, "True Negative", limit=10)
        
        # Save summary
        summary = {
            'total_pairs': len(pairs),
            'true_positives': len(true_positives),
            'true_negatives': len(true_negatives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'accuracy': (len(true_positives) + len(true_negatives)) / len(pairs),
            'threshold': threshold,
            'false_positive_cases': false_positives,
            'false_negative_cases': false_negatives
        }
        
        summary_path = os.path.join(output_dir, 'error_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Saved error summary to: {summary_path}")
        print(f"✅ Error visualizations saved to: {output_dir}")
        
        return summary
    
    def _save_pairs(self, pairs, output_dir, category, limit=None):
        """Save image pairs to directory."""
        if limit:
            pairs = pairs[:limit]
        
        if len(pairs) == 0:
            print(f"  No {category} cases found")
            return
        
        print(f"\nSaving {len(pairs)} {category} cases...")
        
        for i, pair_info in enumerate(pairs):
            # Create pair folder
            pair_dir = os.path.join(output_dir, f"pair_{i:03d}_sim{pair_info['similarity']:.3f}")
            os.makedirs(pair_dir, exist_ok=True)
            
            # Copy images
            img1_src = pair_info['image1']
            img2_src = pair_info['image2']
            
            img1_dst = os.path.join(pair_dir, f"img1_{os.path.basename(img1_src)}")
            img2_dst = os.path.join(pair_dir, f"img2_{os.path.basename(img2_src)}")
            
            shutil.copy2(img1_src, img1_dst)
            shutil.copy2(img2_src, img2_dst)
            
            # Create info file
            info_path = os.path.join(pair_dir, 'info.txt')
            with open(info_path, 'w') as f:
                f.write(f"Category: {category}\n")
                f.write(f"Similarity: {pair_info['similarity']:.4f}\n")
                f.write(f"Prediction: {'Same' if pair_info['prediction'] == 1 else 'Different'}\n")
                f.write(f"Actual: {'Same' if pair_info['actual'] == 1 else 'Different'}\n")
                f.write(f"Image 1: {img1_src}\n")
                f.write(f"Image 2: {img2_src}\n")
            
            # Create side-by-side visualization
            self._create_comparison_image(img1_src, img2_src, pair_info, pair_dir, category)
        
        print(f"  ✅ Saved {len(pairs)} {category} pairs to {output_dir}")
    
    def _create_comparison_image(self, img1_path, img2_path, pair_info, output_dir, category):
        """Create side-by-side comparison image with labels."""
        import matplotlib.pyplot as plt
        
        # Load images
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(img1)
        axes[0].set_title('Image 1')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title('Image 2')
        axes[1].axis('off')
        
        # Add info
        pred_text = 'Same Person' if pair_info['prediction'] == 1 else 'Different Person'
        actual_text = 'Same Person' if pair_info['actual'] == 1 else 'Different Person'
        
        fig.suptitle(
            f"{category}\n"
            f"Similarity: {pair_info['similarity']:.4f} | "
            f"Predicted: {pred_text} | "
            f"Actual: {actual_text}",
            fontsize=12,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save
        comparison_path = os.path.join(output_dir, 'comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model errors')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--feedback_file', type=str, required=True,
                        help='JSON file with test pairs')
    parser.add_argument('--output_dir', type=str, default='error_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Similarity threshold for classification')
    parser.add_argument('--features_list', type=str, default=None,
                        help='Optional features.list to speed up evaluation')
    parser.add_argument('--max_fp', type=int, default=None,
                        help='Max false positives to save (default: all)')
    parser.add_argument('--max_fn', type=int, default=None,
                        help='Max false negatives to save (default: all)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU mode')
    
    args = parser.parse_args()
    
    # Device
    device = 'cpu' if args.force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Visualize errors
    visualizer = ErrorVisualizer(args.model, device, features_list=args.features_list)
    summary = visualizer.visualize_errors(
        args.feedback_file,
        args.output_dir,
        args.threshold,
        max_fp=args.max_fp,
        max_fn=args.max_fn
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Visualization Complete!")
    print(f"{'='*70}")
    print(f"\nCheck the following directories:")
    print(f"  - False Positives: {os.path.join(args.output_dir, 'false_positives')}")
    print(f"  - False Negatives: {os.path.join(args.output_dir, 'false_negatives')}")
    print(f"  - True Positives (samples): {os.path.join(args.output_dir, 'true_positives')}")
    print(f"  - True Negatives (samples): {os.path.join(args.output_dir, 'true_negatives')}")


if __name__ == '__main__':
    main()
