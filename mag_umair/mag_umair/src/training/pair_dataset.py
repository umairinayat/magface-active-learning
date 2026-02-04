"""Dataset class for loading face pairs with labels."""

import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class FeedbackPairDataset(Dataset):
    """
    Dataset for feedback pairs.
    
    Each sample returns: (image1, image2, label)
    - image1, image2: torch tensors of shape (3, 112, 112)
    - label: 1 (same person) or 0 (different person)
    
    Expected JSON format:
    [
        {"image1": "path/to/img1.jpg", "image2": "path/to/img2.jpg", "label": 1},
        {"image1": "path/to/img3.jpg", "image2": "path/to/img4.jpg", "label": 0},
        ...
    ]
    """
    
    def __init__(self, pairs_file: str, base_dir: Optional[str] = None):
        """
        Args:
            pairs_file: Path to JSON file with pairs
            base_dir: Base directory for relative image paths (defaults to parent of pairs_file)
        """
        pairs_path = Path(pairs_file)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
        
        with open(pairs_path, 'r') as f:
            data = json.load(f)
        
        # Handle both formats: {"pairs": [...]} and just [...]
        if isinstance(data, dict) and 'pairs' in data:
            self.pairs = data['pairs']
        elif isinstance(data, list):
            self.pairs = data
        else:
            raise ValueError(f"Invalid JSON format. Expected list or dict with 'pairs' key, got: {type(data)}")
        
        # Set base directory for image paths
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = pairs_path.parent
        
        print(f"Loaded {len(self.pairs)} pairs from {pairs_file}")
        positive = sum(1 for p in self.pairs if p['label'] == 1)
        negative = sum(1 for p in self.pairs if p['label'] == 0)
        print(f"  Positive (same): {positive}")
        print(f"  Negative (different): {negative}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image to tensor."""
        path = Path(image_path)
        
        if path.is_absolute() and path.exists():
            final_path = path
        else:
            # Search up from base_dir to find the image
            final_path = None
            search_dir = self.base_dir.resolve()
            
            # Try up to 6 levels up
            for _ in range(6):
                candidate = search_dir / path
                if candidate.exists():
                    final_path = candidate
                    break
                
                # Move up one level
                parent = search_dir.parent
                if parent == search_dir:  # Reached root
                    break
                search_dir = parent
            
            if final_path is None:
                raise ValueError(f"Could not load image: {image_path}. Base dir: {self.base_dir}")
        
        # Load image
        img = cv2.imread(str(final_path))
        if img is None:
            raise ValueError(f"Could not load image: {final_path}")
        
        # Resize to 112x112 if needed
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        
        # Convert to float and normalize to [0, 1]
        # NOTE: MagFace uses [0, 1] range, NOT [-1, 1]
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        return torch.from_numpy(img)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        
        img1 = self._load_image(pair['image1'])
        img2 = self._load_image(pair['image2'])
        label = torch.tensor(pair['label'], dtype=torch.float32)
        
        return img1, img2, label
