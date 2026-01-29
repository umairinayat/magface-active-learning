import numpy as np
from typing import Dict, Any, List, Optional, Union, Protocol
from pathlib import Path
from PIL import Image
import cv2


class ImageClassifier(Protocol):
    def predict(self, image) -> Dict[str, Any]: ...


class FaceQualityFilter(Protocol):
    def check_face(self, face_data: Dict[str, Any]) -> Dict[str, Any]: ...


# pipeline to filter images
class FilterPipeline:

    
    def __init__(
        self,
        image_classifier: Optional[ImageClassifier] = None,
        quality_filter: Optional[FaceQualityFilter] = None,
        embedder=None
    ):
        self.image_classifier = image_classifier
        self.quality_filter = quality_filter
        self.embedder = embedder
    
    def _load_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def filter(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        result = {
            'is_valid': False,
            'reason': None,
            'faces': []
        }
        
        if self.image_classifier is not None:
            classifier_result = self.image_classifier.predict(image)
            if classifier_result.get('is_anime') or classifier_result.get('is_cartoon'):
                result['reason'] = 'anime'
                result['classifier_result'] = classifier_result
                return result
        
        if self.embedder is None:
            raise RuntimeError("Embedder not set.")
        
        img = self._load_image(image)
        faces = self.embedder.app.get(img)
        
        if not faces:
            result['reason'] = 'no_face'
            return result
        
        face_results = []
        valid_faces = 0
        
        for i, face in enumerate(faces):
            face_data = {
                'bbox': face.bbox.tolist(),
                'det_score': float(face.det_score),
                'embedding': face.embedding
            }
            
            if self.quality_filter is not None:
                quality_result = self.quality_filter.check_face(face_data)
                passed = quality_result['passed']
            else:
                quality_result = {'passed': True, 'reasons': []}
                passed = True
            
            face_result = {
                'face_index': i,
                'passed': passed,
                'det_score': float(face.det_score),
                'face_size': (int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1])),
                'embedding_norm': float(np.linalg.norm(face.embedding)),
                'bbox': face.bbox.tolist(),
                'reasons': quality_result.get('reasons', []),
                'embedding': face.embedding
            }
            
            if passed:
                valid_faces += 1
            
            face_results.append(face_result)
        
        result['faces'] = face_results
        
        if valid_faces > 0:
            result['is_valid'] = True
        else:
            result['reason'] = 'low_quality'
        
        return result
