#!/usr/bin/env python3
"""
Compute MagFace baseline metrics for face verification/identification.
Computes: TAR@FAR, ROC-AUC, Accuracy, Rank-1 Identification Rate
"""

import numpy as np
from collections import defaultdict
import argparse
from sklearn.metrics import roc_curve, auc
import os

def load_features(feat_file):
    """Load features from MagFace output file."""
    features = {}
    with open(feat_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_path = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            features[img_path] = embedding
    return features

def get_identity_from_path(path):
    """Extract identity from path like .../Christian_Bale/Christian_Bale_11936.png"""
    basename = os.path.basename(path)
    # Identity is filename without the last _XXXXX.png
    parts = basename.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return basename

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def compute_verification_metrics(features, facescrub_only=True):
    """
    Compute face verification metrics.
    Returns TAR@FAR values, AUC, and accuracy.
    """
    # Separate facescrub (probe) and megaface (distractor) images
    facescrub_features = {}
    megaface_features = {}
    
    for path, emb in features.items():
        if 'facescrub_images' in path:
            facescrub_features[path] = emb
        elif 'megaface_images' in path:
            megaface_features[path] = emb
    
    print(f"Facescrub images: {len(facescrub_features)}")
    print(f"Megaface images: {len(megaface_features)}")
    
    # Group facescrub by identity
    identity_to_images = defaultdict(list)
    for path, emb in facescrub_features.items():
        identity = get_identity_from_path(path)
        identity_to_images[identity].append((path, emb))
    
    print(f"Unique identities in facescrub: {len(identity_to_images)}")
    
    # Generate positive pairs (same identity) and negative pairs (different identity)
    positive_scores = []
    negative_scores = []
    
    identities = list(identity_to_images.keys())
    
    print("\nGenerating verification pairs...")
    
    # Positive pairs: same identity
    for identity, images in identity_to_images.items():
        if len(images) >= 2:
            for i in range(len(images)):
                for j in range(i + 1, min(len(images), i + 3)):  # Limit pairs per identity
                    score = cosine_similarity(images[i][1], images[j][1])
                    positive_scores.append(score)
    
    # Negative pairs: different identities (sample to balance)
    num_neg_needed = len(positive_scores) * 2
    np.random.seed(42)
    
    for _ in range(num_neg_needed):
        id1, id2 = np.random.choice(len(identities), 2, replace=False)
        imgs1 = identity_to_images[identities[id1]]
        imgs2 = identity_to_images[identities[id2]]
        
        img1 = imgs1[np.random.randint(len(imgs1))]
        img2 = imgs2[np.random.randint(len(imgs2))]
        
        score = cosine_similarity(img1[1], img2[1])
        negative_scores.append(score)
    
    # Add megaface distractors as negative pairs
    if len(megaface_features) > 0:
        megaface_list = list(megaface_features.items())
        facescrub_list = list(facescrub_features.items())
        
        num_distractor_pairs = min(len(positive_scores) * 2, 10000)
        for _ in range(num_distractor_pairs):
            fs_idx = np.random.randint(len(facescrub_list))
            mf_idx = np.random.randint(len(megaface_list))
            
            score = cosine_similarity(facescrub_list[fs_idx][1], megaface_list[mf_idx][1])
            negative_scores.append(score)
    
    print(f"Positive pairs: {len(positive_scores)}")
    print(f"Negative pairs: {len(negative_scores)}")
    
    # Compute metrics
    labels = np.array([1] * len(positive_scores) + [0] * len(negative_scores))
    scores = np.array(positive_scores + negative_scores)
    
    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # TAR@FAR values
    tar_at_far = {}
    for target_far in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        idx = np.argmin(np.abs(fpr - target_far))
        tar_at_far[target_far] = tpr[idx]
    
    # Best accuracy
    accuracies = []
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        acc = np.mean(preds == labels)
        accuracies.append(acc)
    best_acc = max(accuracies)
    best_thresh = thresholds[np.argmax(accuracies)]
    
    return {
        'roc_auc': roc_auc,
        'tar_at_far': tar_at_far,
        'best_accuracy': best_acc,
        'best_threshold': best_thresh,
        'num_positive_pairs': len(positive_scores),
        'num_negative_pairs': len(negative_scores),
        'mean_positive_score': np.mean(positive_scores),
        'mean_negative_score': np.mean(negative_scores),
        'std_positive_score': np.std(positive_scores),
        'std_negative_score': np.std(negative_scores),
    }

def compute_identification_metrics(features):
    """
    Compute Rank-1 identification accuracy.
    For each probe, find if the closest match is the same identity.
    """
    # Separate facescrub (probe) images
    facescrub_features = {}
    megaface_features = {}
    
    for path, emb in features.items():
        if 'facescrub_images' in path:
            facescrub_features[path] = emb
        elif 'megaface_images' in path:
            megaface_features[path] = emb
    
    # Group facescrub by identity
    identity_to_images = defaultdict(list)
    for path, emb in facescrub_features.items():
        identity = get_identity_from_path(path)
        identity_to_images[identity].append((path, emb))
    
    # Only use identities with at least 2 images
    valid_identities = {k: v for k, v in identity_to_images.items() if len(v) >= 2}
    print(f"\nIdentities with >= 2 images for identification: {len(valid_identities)}")
    
    # Rank-1 identification within facescrub gallery
    correct = 0
    total = 0
    
    all_gallery = []
    for identity, images in valid_identities.items():
        for path, emb in images:
            all_gallery.append((path, emb, identity))
    
    print(f"Gallery size: {len(all_gallery)}")
    
    for probe_path, probe_emb, probe_identity in all_gallery:
        best_score = -1
        best_identity = None
        
        for gallery_path, gallery_emb, gallery_identity in all_gallery:
            if gallery_path == probe_path:
                continue
            
            score = cosine_similarity(probe_emb, gallery_emb)
            if score > best_score:
                best_score = score
                best_identity = gallery_identity
        
        if best_identity == probe_identity:
            correct += 1
        total += 1
    
    rank1_acc = correct / total if total > 0 else 0
    
    return {
        'rank1_accuracy': rank1_acc,
        'correct': correct,
        'total': total
    }

def main():
    parser = argparse.ArgumentParser(description='Compute MagFace baseline metrics')
    parser.add_argument('--features', type=str, required=True, help='Path to features.list file')
    parser.add_argument('--output', type=str, default='baseline_metrics.txt', help='Output file for metrics')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MagFace Baseline Metrics Computation")
    print("=" * 60)
    
    print(f"\nLoading features from: {args.features}")
    features = load_features(args.features)
    print(f"Loaded {len(features)} embeddings")
    
    # Compute verification metrics
    print("\n" + "=" * 60)
    print("VERIFICATION METRICS")
    print("=" * 60)
    verification = compute_verification_metrics(features)
    
    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'ROC-AUC':<30} {verification['roc_auc']:.6f}")
    print(f"{'Best Accuracy':<30} {verification['best_accuracy']*100:.4f}%")
    print(f"{'Best Threshold':<30} {verification['best_threshold']:.6f}")
    print(f"{'Mean Positive Score':<30} {verification['mean_positive_score']:.6f}")
    print(f"{'Mean Negative Score':<30} {verification['mean_negative_score']:.6f}")
    print(f"{'Std Positive Score':<30} {verification['std_positive_score']:.6f}")
    print(f"{'Std Negative Score':<30} {verification['std_negative_score']:.6f}")
    
    print(f"\n{'TAR @ FAR':<20} {'Value':<20}")
    print("-" * 40)
    for far, tar in sorted(verification['tar_at_far'].items()):
        print(f"TAR@FAR={far:<12} {tar*100:.4f}%")
    
    # Compute identification metrics
    print("\n" + "=" * 60)
    print("IDENTIFICATION METRICS")
    print("=" * 60)
    identification = compute_identification_metrics(features)
    
    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Rank-1 Accuracy':<30} {identification['rank1_accuracy']*100:.4f}%")
    print(f"{'Correct / Total':<30} {identification['correct']} / {identification['total']}")
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MAGFACE BASELINE METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("VERIFICATION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"ROC-AUC: {verification['roc_auc']:.6f}\n")
        f.write(f"Best Accuracy: {verification['best_accuracy']*100:.4f}%\n")
        f.write(f"Best Threshold: {verification['best_threshold']:.6f}\n")
        f.write(f"Mean Positive Score: {verification['mean_positive_score']:.6f}\n")
        f.write(f"Mean Negative Score: {verification['mean_negative_score']:.6f}\n")
        f.write(f"Std Positive Score: {verification['std_positive_score']:.6f}\n")
        f.write(f"Std Negative Score: {verification['std_negative_score']:.6f}\n\n")
        
        f.write("TAR @ FAR\n")
        f.write("-" * 40 + "\n")
        for far, tar in sorted(verification['tar_at_far'].items()):
            f.write(f"TAR@FAR={far}: {tar*100:.4f}%\n")
        
        f.write("\nIDENTIFICATION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Rank-1 Accuracy: {identification['rank1_accuracy']*100:.4f}%\n")
        f.write(f"Correct / Total: {identification['correct']} / {identification['total']}\n")
    
    print(f"\n\nMetrics saved to: {args.output}")
    print("=" * 60)

if __name__ == '__main__':
    main()
