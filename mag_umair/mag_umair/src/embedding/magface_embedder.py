import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
from collections import OrderedDict
import cv2

from repos.MagFace.models import iresnet


class MagFaceEmbedder:
    
    def __init__(
        self,
        model_path: str,
        arch: str = "iresnet100",
        embedding_size: int = 512,
        device: str = "cpu"
    ):
        self.device = device
        self.embedding_size = embedding_size
        self.arch = arch
        
        # Build model
        if arch == "iresnet100":
            self.model = iresnet.iresnet100(pretrained=False, num_classes=embedding_size)
        elif arch == "iresnet50":
            self.model = iresnet.iresnet50(pretrained=False, num_classes=embedding_size)
        elif arch == "iresnet18":
            self.model = iresnet.iresnet18(pretrained=False, num_classes=embedding_size)
        else:
            raise ValueError(f"Unknown architecture: {arch}. Use iresnet100, iresnet50, or iresnet18")
        
        # Load weights
        self._load_weights(model_path)
        
        # Set to eval mode
        self.model.eval()
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print(f"MagFaceEmbedder initialized: arch={arch}, device={device}")
    
    def _load_weights(self, model_path: str):
        """Load pretrained weights from checkpoint."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Clean state dict keys (handle various training configurations)
        # MagFace checkpoint uses: features.module.{layer} format
        cleaned = OrderedDict()
        model_dict = self.model.state_dict()
        
        for k, v in state_dict.items():
            # Handle DDP training prefixes: features.module.X -> X
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
            
            # Skip classifier head (fc.weight with shape [num_classes, 512])
            # but keep the embedding fc layer (fc.weight with shape [512, 25088])
            if new_k == 'fc.weight' and v.shape[0] != self.embedding_size:
                continue
            
            # Only load if shape matches
            if new_k in model_dict and v.shape == model_dict[new_k].shape:
                cleaned[new_k] = v
        
        # Load weights
        self.model.load_state_dict(cleaned, strict=False)
        print(f"Loaded {len(cleaned)} weight tensors (excluding fc layer)")
    
    def _preprocess(self, aligned_face: np.ndarray) -> torch.Tensor:

        # Ensure correct size
        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))
        
        # BGR -> RGB (optional, MagFace was trained on BGR based on gen_feat.py)
        # Actually checking gen_feat.py: uses cv2.imread which is BGR, no conversion
        # So we keep BGR
        
        # Normalize to [0, 1] then to tensor
        # MagFace gen_feat.py uses: mean=[0., 0., 0.], std=[1., 1., 1.]
        # Which means just divide by 255
        img = aligned_face.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        tensor = torch.from_numpy(img).unsqueeze(0)
        
        if self.device == "cuda" and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        return tensor
    
    def get_embedding(self, aligned_face: np.ndarray) -> Tuple[np.ndarray, float]:
        tensor = self._preprocess(aligned_face)
        
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = embedding.cpu().numpy()[0]
        
        # Magnitude = quality score (MagFace's unique feature)
        magnitude = float(np.linalg.norm(embedding))
        
        # L2 normalize for cosine similarity comparison
        if magnitude > 0:
            normalized_embedding = embedding / magnitude
        else:
            normalized_embedding = embedding
        
        return normalized_embedding, magnitude
    
    def get_embeddings_batch(
        self, 
        aligned_faces: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, float]]:

        if not aligned_faces:
            return []
        
        # Stack into batch
        batch = torch.cat([self._preprocess(face) for face in aligned_faces], dim=0)
        
        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = embeddings.cpu().numpy()
        
        results = []
        for emb in embeddings:
            magnitude = float(np.linalg.norm(emb))
            if magnitude > 0:
                normalized = emb / magnitude
            else:
                normalized = emb
            results.append((normalized, magnitude))
        
        return results
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_size
