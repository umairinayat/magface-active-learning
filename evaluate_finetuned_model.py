#!/usr/bin/env python
"""
Evaluate Fine-Tuned Model on Feedback Pairs

This script evaluates:
1. Pair classification accuracy (same/different)
2. Distance distribution analysis
3. ROC curve and optimal threshold
4. Comparison with pretrained model
"""
import sys
sys.path.insert(0, 'MagFace_repo')

import os
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms as T
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns

from MagFace_repo.models.iresnet import iresnet100


class ModelEvaluator:
    """
    Evaluate fine-tuned model on feedback pairs.
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: Path to model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Load model
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
        
        # Transform
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor()
            # MagFace expects [0, 1] range, NO mean/std normalization
        ])
        
        print("✅ Model loaded")
    
    def extract_embedding(self, image_path):
        """Extract embedding from image."""
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(img_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0]
    
    def compute_distance(self, emb1, emb2):
        """Compute Euclidean distance between embeddings."""
        return np.linalg.norm(emb1 - emb2)
    
    def compute_cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings."""
        return np.dot(emb1, emb2)
    
    def evaluate_pairs(self, pairs_file):
        """
        Evaluate model on feedback pairs.
        
        Returns:
            results: Dict with evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating Model on Feedback Pairs")
        print(f"{'='*70}")
        
        # Load pairs
        with open(pairs_file, 'r') as f:
            data = json.load(f)
        pairs = data['pairs']
        
        print(f"Loaded {len(pairs)} pairs")
        
        # Compute distances
        distances = []
        similarities = []
        labels = []
        
        print("\nComputing embeddings and distances...")
        for pair in tqdm(pairs):
            # Extract embeddings
            emb1 = self.extract_embedding(pair['image1'])
            emb2 = self.extract_embedding(pair['image2'])
            
            # Compute metrics
            dist = self.compute_distance(emb1, emb2)
            sim = self.compute_cosine_similarity(emb1, emb2)
            
            distances.append(dist)
            similarities.append(sim)
            labels.append(pair['label'])
        
        distances = np.array(distances)
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Statistics
        results = {
            'total_pairs': len(pairs),
            'positive_pairs': np.sum(labels == 1),
            'negative_pairs': np.sum(labels == 0),
            'distances': distances,
            'similarities': similarities,
            'labels': labels
        }
        
        # Distance statistics
        same_distances = distances[labels == 1]
        diff_distances = distances[labels == 0]
        
        results['same_person'] = {
            'mean_distance': np.mean(same_distances),
            'std_distance': np.std(same_distances),
            'min_distance': np.min(same_distances),
            'max_distance': np.max(same_distances),
            'mean_similarity': np.mean(similarities[labels == 1]),
        }
        
        results['different_person'] = {
            'mean_distance': np.mean(diff_distances),
            'std_distance': np.std(diff_distances),
            'min_distance': np.min(diff_distances),
            'max_distance': np.max(diff_distances),
            'mean_similarity': np.mean(similarities[labels == 0]),
        }
        
        # Find optimal threshold
        results['optimal_threshold'] = self._find_optimal_threshold(
            distances, labels
        )
        
        # Compute accuracy at different thresholds
        results['threshold_analysis'] = self._threshold_analysis(
            distances, labels
        )
        
        # ROC curve
        results['roc'] = self._compute_roc(distances, labels)
        
        return results
    
    def _find_optimal_threshold(self, distances, labels):
        """Find optimal similarity threshold using ROC curve."""
        # Use similarities instead of distances
        # For similarity: higher = more similar (same person)
        similarities = 1 - distances / 2  # Convert distance to similarity approximation
        
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        
        # Find threshold that maximizes TPR - FPR
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'threshold': optimal_threshold,
            'tpr': tpr[optimal_idx],
            'fpr': fpr[optimal_idx],
            'accuracy': (tpr[optimal_idx] + (1 - fpr[optimal_idx])) / 2
        }
    
    def _threshold_analysis(self, distances, labels):
        """Analyze accuracy at different similarity thresholds."""
        # Use similarity thresholds from -1.0 to 1.0
        thresholds = np.arange(-1.0, 1.0, 0.05)
        results = []
        
        # Get similarities from stored results
        similarities = 1 - distances / 2  # Approximate conversion
        
        for threshold in thresholds:
            # Predict: similarity > threshold → same person (1)
            predictions = (similarities > threshold).astype(int)
            
            # Metrics
            accuracy = np.mean(predictions == labels)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            })
        
        return results
    
    def _compute_roc(self, distances, labels):
        """Compute ROC curve using similarities."""
        # Use similarities (higher = positive)
        similarities = 1 - distances / 2  # Approximate conversion
        
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    def print_results(self, results):
        """Print evaluation results."""
        print(f"\n{'='*70}")
        print(f"Evaluation Results")
        print(f"{'='*70}")
        
        print(f"\nDataset Statistics:")
        print(f"  Total pairs: {results['total_pairs']}")
        print(f"  Positive (same person): {results['positive_pairs']}")
        print(f"  Negative (different person): {results['negative_pairs']}")
        
        print(f"\nDistance Statistics:")
        print(f"  Same person:")
        print(f"    Mean distance: {results['same_person']['mean_distance']:.4f}")
        print(f"    Std distance: {results['same_person']['std_distance']:.4f}")
        print(f"    Range: [{results['same_person']['min_distance']:.4f}, {results['same_person']['max_distance']:.4f}]")
        print(f"    Mean similarity: {results['same_person']['mean_similarity']:.4f}")
        
        print(f"  Different person:")
        print(f"    Mean distance: {results['different_person']['mean_distance']:.4f}")
        print(f"    Std distance: {results['different_person']['std_distance']:.4f}")
        print(f"    Range: [{results['different_person']['min_distance']:.4f}, {results['different_person']['max_distance']:.4f}]")
        print(f"    Mean similarity: {results['different_person']['mean_similarity']:.4f}")
        
        print(f"\nOptimal Threshold:")
        opt = results['optimal_threshold']
        print(f"  Threshold: {opt['threshold']:.4f}")
        print(f"  True Positive Rate: {opt['tpr']:.4f}")
        print(f"  False Positive Rate: {opt['fpr']:.4f}")
        print(f"  Accuracy: {opt['accuracy']:.4f}")
        
        print(f"\nROC AUC Score: {results['roc']['auc']:.4f}")
        
        # Best threshold from analysis
        best_result = max(results['threshold_analysis'], key=lambda x: x['accuracy'])
        print(f"\nBest Accuracy Threshold:")
        print(f"  Threshold: {best_result['threshold']:.4f}")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  Precision: {best_result['precision']:.4f}")
        print(f"  Recall: {best_result['recall']:.4f}")
        print(f"  F1-Score: {best_result['f1']:.4f}")
    
    def plot_results(self, results, output_dir='evaluation_plots'):
        """Create visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        distances = results['distances']
        labels = results['labels']
        
        # Plot 1: Distance distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(distances[labels == 1], bins=50, alpha=0.7, label='Same person', color='green')
        plt.hist(distances[labels == 0], bins=50, alpha=0.7, label='Different person', color='red')
        plt.axvline(results['optimal_threshold']['threshold'], color='blue', 
                   linestyle='--', label=f"Optimal threshold: {results['optimal_threshold']['threshold']:.3f}")
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.title('Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Similarity distribution
        plt.subplot(1, 2, 2)
        similarities = results['similarities']
        plt.hist(similarities[labels == 1], bins=50, alpha=0.7, label='Same person', color='green')
        plt.hist(similarities[labels == 0], bins=50, alpha=0.7, label='Different person', color='red')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Similarity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=150)
        print(f"\n✅ Saved: {output_dir}/distance_distribution.png")
        plt.close()
        
        # Plot 3: ROC Curve
        plt.figure(figsize=(8, 8))
        roc = results['roc']
        plt.plot(roc['fpr'], roc['tpr'], color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        print(f"✅ Saved: {output_dir}/roc_curve.png")
        plt.close()
        
        # Plot 4: Threshold analysis
        plt.figure(figsize=(12, 5))
        
        threshold_results = results['threshold_analysis']
        thresholds = [r['threshold'] for r in threshold_results]
        accuracies = [r['accuracy'] for r in threshold_results]
        precisions = [r['precision'] for r in threshold_results]
        recalls = [r['recall'] for r in threshold_results]
        f1s = [r['f1'] for r in threshold_results]
        
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2)
        plt.plot(thresholds, precisions, 's-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, '^-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1s, 'd-', label='F1-Score', linewidth=2)
        plt.axvline(results['optimal_threshold']['threshold'], color='red', 
                   linestyle='--', alpha=0.7, label='Optimal')
        plt.xlabel('Similarity Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Confusion matrix at optimal threshold
        plt.subplot(1, 2, 2)
        opt_threshold = results['optimal_threshold']['threshold']
        similarities = results['similarities']
        predictions = (similarities > opt_threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Different', 'Same'],
                   yticklabels=['Different', 'Same'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (threshold={opt_threshold:.3f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=150)
        print(f"✅ Saved: {output_dir}/threshold_analysis.png")
        plt.close()


def compare_models(pretrained_path, finetuned_path, pairs_file, device='cpu'):
    """
    Compare pretrained vs fine-tuned model.
    """
    print(f"\n{'='*70}")
    print(f"Comparing Pretrained vs Fine-Tuned Models")
    print(f"{'='*70}")
    
    # Evaluate pretrained
    print(f"\n[1/2] Evaluating Pretrained Model...")
    evaluator_pretrained = ModelEvaluator(pretrained_path, device)
    results_pretrained = evaluator_pretrained.evaluate_pairs(pairs_file)
    
    # Evaluate fine-tuned
    print(f"\n[2/2] Evaluating Fine-Tuned Model...")
    evaluator_finetuned = ModelEvaluator(finetuned_path, device)
    results_finetuned = evaluator_finetuned.evaluate_pairs(pairs_file)
    
    # Comparison
    print(f"\n{'='*70}")
    print(f"Comparison Results")
    print(f"{'='*70}")
    
    print(f"\nPretrained Model:")
    print(f"  Same person mean distance: {results_pretrained['same_person']['mean_distance']:.4f}")
    print(f"  Different person mean distance: {results_pretrained['different_person']['mean_distance']:.4f}")
    print(f"  Separation: {results_pretrained['different_person']['mean_distance'] - results_pretrained['same_person']['mean_distance']:.4f}")
    print(f"  ROC AUC: {results_pretrained['roc']['auc']:.4f}")
    print(f"  Best Accuracy: {max(r['accuracy'] for r in results_pretrained['threshold_analysis']):.4f}")
    
    print(f"\nFine-Tuned Model:")
    print(f"  Same person mean distance: {results_finetuned['same_person']['mean_distance']:.4f}")
    print(f"  Different person mean distance: {results_finetuned['different_person']['mean_distance']:.4f}")
    print(f"  Separation: {results_finetuned['different_person']['mean_distance'] - results_finetuned['same_person']['mean_distance']:.4f}")
    print(f"  ROC AUC: {results_finetuned['roc']['auc']:.4f}")
    print(f"  Best Accuracy: {max(r['accuracy'] for r in results_finetuned['threshold_analysis']):.4f}")
    
    # Improvement
    improvement = {
        'separation': (results_finetuned['different_person']['mean_distance'] - results_finetuned['same_person']['mean_distance']) - 
                     (results_pretrained['different_person']['mean_distance'] - results_pretrained['same_person']['mean_distance']),
        'auc': results_finetuned['roc']['auc'] - results_pretrained['roc']['auc'],
        'accuracy': max(r['accuracy'] for r in results_finetuned['threshold_analysis']) - 
                   max(r['accuracy'] for r in results_pretrained['threshold_analysis'])
    }
    
    print(f"\nImprovement:")
    print(f"  Separation: {improvement['separation']:+.4f} ({improvement['separation']/max(0.001, results_pretrained['different_person']['mean_distance'] - results_pretrained['same_person']['mean_distance'])*100:+.1f}%)")
    print(f"  ROC AUC: {improvement['auc']:+.4f} ({improvement['auc']/results_pretrained['roc']['auc']*100:+.1f}%)")
    print(f"  Accuracy: {improvement['accuracy']:+.4f} ({improvement['accuracy']*100:+.1f}%)")
    
    return results_pretrained, results_finetuned, improvement


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model')
    parser.add_argument('--finetuned', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model for comparison')
    parser.add_argument('--feedback_file', type=str, default='feedback_pairs.json',
                        help='JSON file with feedback pairs')
    parser.add_argument('--output_dir', type=str, default='evaluation_plots',
                        help='Output directory for plots')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU mode')
    
    args = parser.parse_args()
    
    # Device
    device = 'cpu' if args.force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.pretrained:
        # Compare models
        results_pre, results_ft, improvement = compare_models(
            args.pretrained,
            args.finetuned,
            args.feedback_file,
            device
        )
        
        # Plot both
        print(f"\nCreating plots for pretrained model...")
        evaluator_pre = ModelEvaluator(args.pretrained, device)
        evaluator_pre.plot_results(results_pre, os.path.join(args.output_dir, 'pretrained'))
        
        print(f"\nCreating plots for fine-tuned model...")
        evaluator_ft = ModelEvaluator(args.finetuned, device)
        evaluator_ft.plot_results(results_ft, os.path.join(args.output_dir, 'finetuned'))
    else:
        # Evaluate single model
        evaluator = ModelEvaluator(args.finetuned, device)
        results = evaluator.evaluate_pairs(args.feedback_file)
        evaluator.print_results(results)
        evaluator.plot_results(results, args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
