"""MagFace Trainer for fine-tuning on feedback pairs."""

import sys
from pathlib import Path

# Add parent folder to path for MagFace_repo import
_parent_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from tqdm import tqdm

from MagFace_repo.models import iresnet
from .contrastive_loss import ContrastiveLoss
from .pair_dataset import FeedbackPairDataset


class MagFaceTrainer:
    """
    Trainer for fine-tuning MagFace on feedback pairs.
    
    Uses Contrastive Loss to:
    - Pull same-person embeddings closer
    - Push different-person embeddings apart
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        arch: str = "iresnet100",
        embedding_size: int = 512,
        device: str = "cpu"
    ):
        """
        Args:
            checkpoint_path: Path to pretrained MagFace checkpoint
            arch: Model architecture (iresnet100, iresnet50, iresnet18)
            embedding_size: Embedding dimension (default 512)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.embedding_size = embedding_size
        
        # Build model
        print(f"Building {arch} model...")
        if arch == "iresnet100":
            self.model = iresnet.iresnet100(pretrained=False, num_classes=embedding_size)
        elif arch == "iresnet50":
            self.model = iresnet.iresnet50(pretrained=False, num_classes=embedding_size)
        elif arch == "iresnet18":
            self.model = iresnet.iresnet18(pretrained=False, num_classes=embedding_size)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Load pretrained weights
        self._load_checkpoint(checkpoint_path)
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print(f"MagFaceTrainer initialized: arch={arch}, device={device}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights with proper key cleaning."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Clean state dict keys (handle DDP training prefixes)
        cleaned = OrderedDict()
        model_dict = self.model.state_dict()
        
        for k, v in state_dict.items():
            # Handle various prefixes
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
            
            # Skip classifier head (fc.weight with wrong shape)
            if new_k == 'fc.weight' and v.shape[0] != self.embedding_size:
                continue
            
            # Only load if shape matches
            if new_k in model_dict and v.shape == model_dict[new_k].shape:
                cleaned[new_k] = v
        
        self.model.load_state_dict(cleaned, strict=False)
        print(f"Loaded {len(cleaned)} weight tensors")
    
    def finetune(
        self,
        train_pairs_file: str,
        val_pairs_file: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.0001,
        margin: float = 1.0,
        output_dir: str = "checkpoints",
        similarity_threshold: float = 0.4
    ) -> Dict[str, Any]:
        """
        Fine-tune the model on feedback pairs.
        
        Args:
            train_pairs_file: Path to training pairs JSON
            val_pairs_file: Path to validation pairs JSON (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for Adam optimizer
            margin: Contrastive loss margin
            output_dir: Directory to save checkpoints
            similarity_threshold: Threshold for accuracy calculation
        
        Returns:
            Dictionary with training history
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        train_dataset = FeedbackPairDataset(train_pairs_file)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        val_loader = None
        if val_pairs_file:
            val_dataset = FeedbackPairDataset(val_pairs_file)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Setup training
        criterion = ContrastiveLoss(margin=margin)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'best_accuracy': 0.0,
            'best_epoch': 0
        }
        
        print(f"\n{'='*60}")
        print(f"Starting fine-tuning:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Margin: {margin}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion, similarity_threshold)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                print(f"Epoch {epoch}/{epochs}: Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2%}")
                
                # Save best model
                if val_acc > history['best_accuracy']:
                    history['best_accuracy'] = val_acc
                    history['best_epoch'] = epoch
                    best_path = output_path / "magface_finetuned_best.pth"
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  → Saved best model: {best_path}")
            else:
                print(f"Epoch {epoch}/{epochs}: Train Loss = {train_loss:.4f}")
            
            # Save epoch checkpoint
            epoch_path = output_path / f"magface_finetuned_epoch{epoch}.pth"
            torch.save(self.model.state_dict(), epoch_path)
        
        # Save final model
        final_path = output_path / "magface_finetuned_final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"\nFinal model saved: {final_path}")
        
        if history['best_accuracy'] > 0:
            print(f"Best accuracy: {history['best_accuracy']:.2%} (epoch {history['best_epoch']})")
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for img1, img2, labels in tqdm(dataloader, desc="Training", leave=False):
            if self.device == "cuda":
                img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
            
            # Forward pass
            emb1 = self.model(img1)
            emb2 = self.model(img2)
            
            # L2 normalize embeddings
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            
            # Compute loss
            loss = criterion(emb1, emb2, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate(self, dataloader: DataLoader, criterion: nn.Module, threshold: float) -> Tuple[float, float]:
        """Validate model and compute accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(dataloader, desc="Validating", leave=False):
                if self.device == "cuda":
                    img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
                
                # Forward pass
                emb1 = self.model(img1)
                emb2 = self.model(img2)
                
                # L2 normalize
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)
                
                # Compute loss
                loss = criterion(emb1, emb2, labels)
                total_loss += loss.item()
                
                # Compute accuracy using cosine similarity
                similarities = (emb1 * emb2).sum(dim=1)  # Dot product = cosine sim for normalized vectors
                predictions = (similarities > threshold).float()
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def evaluate(self, pairs_file: str, threshold: float = 0.4) -> Dict[str, Any]:
        """
        Evaluate model on pairs and return metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, etc.
        """
        dataset = FeedbackPairDataset(pairs_file)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        self.model.eval()
        
        all_similarities = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(dataloader, desc="Evaluating"):
                if self.device == "cuda":
                    img1, img2 = img1.cuda(), img2.cuda()
                
                emb1 = self.model(img1)
                emb2 = self.model(img2)
                
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)
                
                similarities = (emb1 * emb2).sum(dim=1).cpu().numpy()
                predictions = (similarities > threshold).astype(int)
                
                all_similarities.extend(similarities.tolist())
                all_labels.extend(labels.numpy().astype(int).tolist())
                all_predictions.extend(predictions.tolist())
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        tn = ((all_predictions == 0) & (all_labels == 0)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        
        accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_pairs': len(all_labels),
            'threshold': threshold
        }
