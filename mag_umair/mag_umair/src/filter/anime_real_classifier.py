import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, TYPE_CHECKING
from PIL import Image
import cv2

if TYPE_CHECKING:
    from ..config_loader import Config

try:
    from imgutils.validate import anime_real_score, anime_real
except ImportError:
    raise ImportError("dghs-imgutils is required. Install with: pip install dghs-imgutils")


class AnimeRealClassifier:
    
    def __init__(self, model_name: str = 'mobilenetv3_v1.4_dist', threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
    
    @classmethod
    def from_config(cls, config: "Config") -> Optional["AnimeRealClassifier"]:
        if not config.filter.anime_classifier.enabled:
            return None
        return cls(
            model_name=config.filter.anime_classifier.model,
            threshold=config.filter.anime_classifier.threshold
        )
    
    def _load_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Union[str, Image.Image]:
        if isinstance(image, (str, Path)):
            return str(image)
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            return Image.fromarray(rgb_image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        img = self._load_image(image)
        
        scores = anime_real_score(img, model_name=self.model_name)
        
        anime_score = scores['anime']
        real_score = scores['real']
        
        is_anime = anime_score > real_score and anime_score > self.threshold
        
        return {
            'is_real': not is_anime,
            'is_cartoon': is_anime,
            'is_anime': is_anime,
            'confidence': float(anime_score if is_anime else real_score),
            'anime_score': float(anime_score),
            'real_score': float(real_score)
        }
    
    def predict_batch(self, images: list) -> list:
        results = []
        for image in images:
            results.append(self.predict(image))
        return results
