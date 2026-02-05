#!/usr/bin/env python
"""
Fine-tune MagFace on User Feedback Pairs

Uses contrastive learning on pairs labeled by users:
- Positive pairs (same person): Pull embeddings closer
- Negative pairs (different person): Push embeddings apart
"""
import sys
sys.path.insert(0, 'MagFace_repo')

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from MagFace_repo.models.iresnet import iresnet100


class FeedbackPairDataset(Dataset):
    """
    Dataset for feedback pairs.
    Each sample: (image1, image2, label)
    - label = 1: same person
    - label = 0: different person
    """
    
    def __init__(self, pairs_file, transform=None):
        """
        Args:
            pairs_file: JSON file with feedback pairs
            transform: Image transform
        """
        with open(pairs_file, 'r') as f:
            data = json.load(f)
        
        self.pairs = data['pairs']
        self.transform = transform
        
        print(f"Loaded {len(self.pairs)} feedback pairs")
        
        # Count positive/negative
        positive = sum(1 for p in self.pairs if p['label'] == 1)
        negative = sum(1 for p in self.pairs if p['label'] == 0)
        print(f"  Positive (same): {positive}")
        print(f"  Negative (different): {negative}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load images
        img1 = cv2.imread(pair['image1'])
        img2 = cv2.imread(pair['image2'])
        
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = torch.tensor(pair['label'], dtype=torch.float32)
        
        return img1, img2, label


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for pair-based learning.
    
    - Positive pairs: Maximize similarity (push towards 1)
    - Negative pairs: Minimize similarity (push towards 0 or below margin)
    
    Uses cosine similarity with threshold 0.4 for consistency.
    """
    
    def __init__(self, margin=0.4):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin  # Similarity threshold
    
    def forward(self, emb1, emb2, label):
        """
        Args:
            emb1: Embeddings from image1 (batch_size, 512) - should be normalized
            emb2: Embeddings from image2 (batch_size, 512) - should be normalized
            label: 1 (same person) or 0 (different person)
        
        Returns:
            loss: Cosine similarity based contrastive loss
        """
        # Cosine similarity (for normalized embeddings, this is just dot product)
        similarity = torch.sum(emb1 * emb2, dim=1)
        
        # Contrastive loss using cosine similarity
        # Same person (label=1): Maximize similarity -> minimize (1 - similarity)
        # Different person (label=0): Minimize similarity -> minimize max(0, similarity - margin)
        loss_positive = label * torch.pow(1 - similarity, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(similarity - self.margin, min=0.0), 2)
        
        loss = torch.mean(loss_positive + loss_negative)
        
        return loss


# Alias for backward compatibility
ContrastiveLoss = CosineSimilarityLoss


def train_epoch(model, dataloader, criterion, optimizer, device, threshold):
    """Train for one epoch.""""""  """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    
    for img1, img2, label in pbar:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        
        # Extract embeddings (train mode)
        emb1 = model(img1)
        emb2 = model(img2)
        
        # Normalize embeddings
        emb1_norm = nn.functional.normalize(emb1, p=2, dim=1)
        emb2_norm = nn.functional.normalize(emb2, p=2, dim=1)
        
        # Compute loss
        loss = criterion(emb1_norm, emb2_norm, label)
        
        # Backward (disabled for forward-pass-only validation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Accuracy using cosine similarity with threshold
        # Use detached embeddings for accuracy (no gradient, stable computation)
        with torch.no_grad():
            # Re-compute in eval mode for accurate metric
            model.eval()
            emb1_eval = model(img1)
            emb2_eval = model(img2)
            emb1_eval = nn.functional.normalize(emb1_eval, p=2, dim=1)
            emb2_eval = nn.functional.normalize(emb2_eval, p=2, dim=1)
            similarity = torch.sum(emb1_eval * emb2_eval, dim=1)
            predicted = (similarity > threshold).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)
            model.train()  # Switch back to train mode
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.3f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, threshold):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc='Validation'):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            
            # Extract embeddings
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Normalize
            emb1 = nn.functional.normalize(emb1, p=2, dim=1)
            emb2 = nn.functional.normalize(emb2, p=2, dim=1)
            
            # Loss
            loss = criterion(emb1, emb2, label)
            total_loss += loss.item()
            
            # Accuracy using cosine similarity
            # For normalized embeddings, cosine similarity = dot product
            similarity = torch.sum(emb1 * emb2, dim=1)
            predicted = (similarity > threshold).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune on feedback pairs')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/magface_finetuned_final.pth',
                        help='Path to initial checkpoint')
    parser.add_argument('--feedback_file', type=str, default='feedback_pairs.json',
                        help='JSON file with feedback pairs')
    parser.add_argument('--output_dir', type=str, default='checkpoints_feedback',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.4,
                        help='Contrastive loss margin')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (ignored if --val_file is set)')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Optional separate validation pairs JSON')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU mode')
    
    args = parser.parse_args()
    
    # Device
    device = 'cpu' if args.force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transform
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor()
        # MagFace expects [0, 1] range, NO mean/std normalization
    ])
    
    # Load datasets
    print(f"\nLoading training pairs from {args.feedback_file}...")
    full_dataset = FeedbackPairDataset(args.feedback_file, transform=transform)

    if args.val_file:
        print(f"Loading validation pairs from {args.val_file}...")
        train_dataset = full_dataset
        val_dataset = FeedbackPairDataset(args.val_file, transform=transform)
    else:
        # Split train/val
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        split_gen = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=split_gen
        )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = iresnet100(num_classes=512)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Extract state_dict and clean keys
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Clean keys: features.module.X -> X
    from collections import OrderedDict
    cleaned = OrderedDict()
    model_dict = model.state_dict()
    
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
    
    model.load_state_dict(cleaned, strict=False)
    print(f"Loaded {len(cleaned)}/{len(model_dict)} weight tensors")
    
    model.to(device)
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting Fine-tuning on Feedback Pairs")
    print(f"{'='*70}")
    
    # Run validation BEFORE training to show baseline
    print("\n[Baseline] Evaluating pretrained model before any training...")
    model.eval()
    baseline_loss, baseline_acc = validate(model, val_loader, criterion, device, threshold=args.margin)
    print(f"[Baseline] Pretrained Val Acc: {baseline_acc:.3f} (should be ~0.98)")
    
    best_val_acc = baseline_acc
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"{'-'*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, threshold=args.margin
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, threshold=args.margin)
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            args.output_dir,
            f'magface_feedback_epoch{epoch+1}.pth'
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_dir, 'magface_feedback_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best model! Val Acc: {val_acc:.3f}")
    
    print(f"\n{'='*70}")
    print(f"✅ Fine-tuning Complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'magface_feedback_best.pth')}")


if __name__ == '__main__':
    main()
