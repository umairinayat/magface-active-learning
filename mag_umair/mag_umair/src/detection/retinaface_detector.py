
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import cv2

from retinaface import RetinaFace


class RetinaFaceDetector:
    def __init__(self, threshold: float = 0.9, allow_upscaling: bool = True):

        self.threshold = threshold
        self.allow_upscaling = allow_upscaling
        # Model loads lazily on first use
    
    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def detect(self, image: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        img = self._load_image(image)
        
        # RetinaFace.detect_faces accepts path or ndarray
        faces = RetinaFace.detect_faces(
            img, 
            threshold=self.threshold,
            allow_upscaling=self.allow_upscaling
        )
        
        # Handle no faces or error
        if not isinstance(faces, dict) or len(faces) == 0:
            return []
        
        results = []
        for face_id, face_data in faces.items():
            landmarks = face_data['landmarks']
            
            # Convert landmarks dict to numpy array (5, 2)
            # CRITICAL: RetinaFace uses VIEWER'S perspective for left/right
            # but ArcFace/MagFace alignment expects SUBJECT'S perspective
            # So we need to swap: left_eye <-> right_eye, mouth_left <-> mouth_right
            lmk = np.array([
                landmarks['right_eye'],   # Subject's left eye (viewer's right)
                landmarks['left_eye'],    # Subject's right eye (viewer's left)
                landmarks['nose'],
                landmarks['mouth_right'], # Subject's left mouth (viewer's right)
                landmarks['mouth_left']   # Subject's right mouth (viewer's left)
            ], dtype=np.float32)
            
            # facial_area is [x1, y1, x2, y2]
            bbox = face_data['facial_area']
            
            results.append({
                'bbox': bbox,
                'landmarks': lmk,
                'det_score': float(face_data['score'])
            })
        
        # Sort by detection score (highest first)
        results.sort(key=lambda x: x['det_score'], reverse=True)
        
        return results
    
    def detect_largest(self, image: Union[str, Path, np.ndarray]) -> Optional[Dict[str, Any]]:

        faces = self.detect(image)
        if not faces:
            return None
        
        # Find largest by bbox area
        def bbox_area(face):
            bbox = face['bbox']
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        return max(faces, key=bbox_area)
