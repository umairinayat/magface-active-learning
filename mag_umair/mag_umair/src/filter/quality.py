import numpy as np
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..config_loader import Config


class QualityFilter:
    
    def __init__(
        self,
        min_det_score: float = 0.5,
        min_face_size: int = 50,
        min_embedding_norm: Optional[float] = None
    ):
        self.min_det_score = min_det_score
        self.min_face_size = min_face_size
        self.min_embedding_norm = min_embedding_norm
    
    @classmethod
    def from_config(cls, config: "Config") -> "QualityFilter":
        return cls(
            min_det_score=config.filter.quality.min_det_score,
            min_face_size=config.filter.quality.min_face_size,
            min_embedding_norm=config.filter.quality.min_embedding_norm
        )
    
    def check_face(self, face: Dict[str, Any]) -> Dict[str, Any]:
        bbox = face.get('bbox', [0, 0, 0, 0])
        det_score = face.get('det_score', 0.0)
        embedding = face.get('embedding')
        
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_size = (int(face_width), int(face_height))
        
        embedding_norm = float(np.linalg.norm(embedding)) if embedding is not None else None
        
        reasons = []
        
        if det_score < self.min_det_score:
            reasons.append(f"low_det_score:{det_score:.2f}<{self.min_det_score}")
        
        if face_width < self.min_face_size or face_height < self.min_face_size:
            reasons.append(f"small_face:{face_size}<{self.min_face_size}x{self.min_face_size}")
        
        if self.min_embedding_norm is not None and embedding_norm is not None:
            if embedding_norm < self.min_embedding_norm:
                reasons.append(f"low_embedding_norm:{embedding_norm:.1f}<{self.min_embedding_norm}")
        
        passed = len(reasons) == 0
        
        return {
            'passed': passed,
            'reasons': reasons,
            'det_score': det_score,
            'face_size': face_size,
            'embedding_norm': embedding_norm,
            'bbox': bbox
        }
    
    def filter_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for face in faces:
            result = self.check_face(face)
            result['face'] = face
            results.append(result)
        return results
