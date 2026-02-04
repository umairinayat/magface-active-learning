"""Contrastive Loss for pair-based metric learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for pair-based learning.
    
    - Positive pairs (same person): Minimize distance
    - Negative pairs (different person): Maximize distance (up to margin)
    
    Formula:
        L = Y * D^2 + (1 - Y) * max(0, margin - D)^2
    
    Where:
        - Y = 1 for same person, 0 for different
        - D = Euclidean distance between embeddings
        - margin = minimum distance for different-person pairs
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            emb1: Embeddings from image1 (batch_size, 512)
            emb2: Embeddings from image2 (batch_size, 512)
            label: 1 (same person) or 0 (different person) - shape (batch_size,)
        
        Returns:
            loss: Contrastive loss (scalar)
        """
        # Compute Euclidean distance
        distance = F.pairwise_distance(emb1, emb2)
        
        # Contrastive loss
        # Same person (label=1): minimize distance -> loss = distance^2
        # Different person (label=0): maximize distance -> loss = max(0, margin - distance)^2
        loss_same = label * distance.pow(2)
        loss_diff = (1 - label) * F.relu(self.margin - distance).pow(2)
        
        loss = (loss_same + loss_diff).mean()
        
        return loss
