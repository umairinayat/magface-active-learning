import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MagFaceConfig:
    weights: str = "models/magface_epoch_00025.pth"
    arch: str = "iresnet100"
    embedding_size: int = 512


@dataclass
class ModelConfig:
    pack: str = "buffalo_l"
    device: str = "cpu"
    allowed_modules: Optional[list] = None
    backend: str = "magface"  # "insightface" or "magface"
    magface: MagFaceConfig = field(default_factory=MagFaceConfig)


@dataclass
class DetectionConfig:
    det_size: tuple = (640, 640)
    det_thresh: float = 0.5
    nms_thresh: float = 0.4
    max_num: int = 0


@dataclass
class RecognitionConfig:
    select_largest: bool = True


@dataclass
class MatchingConfig:
    similarity_threshold: float = 0.4


@dataclass
class IndexConfig:
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection_name: str = "face_embeddings"


@dataclass
class PathsConfig:
    index_dir: str = "data/face_index"
    gallery_dir: str = "data/gallery"
    output_dir: str = "data/output"




@dataclass
class AnimeClassifierConfig:
    enabled: bool = True
    model: str = "mobilenetv3_v1.4_dist"
    threshold: float = 0.6


@dataclass
class FilterConfig:
    enabled: bool = True
    min_det_score: float = 0.5
    min_face_size: int = 16
    min_quality_score: float = 7.0  # MagFace magnitude
    anime_classifier: AnimeClassifierConfig = field(default_factory=AnimeClassifierConfig)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        possible_paths = [
            Path("/workspace/config.yaml"),
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml"
        ]
        for p in possible_paths:
            if p.exists():
                config_path = str(p)
                break
    
    config = Config()
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data:
            if 'model' in data:
                magface_data = data['model'].get('magface', {})
                config.model = ModelConfig(
                    pack=data['model'].get('pack', config.model.pack),
                    device=data['model'].get('device', config.model.device),
                    allowed_modules=data['model'].get('allowed_modules'),
                    backend=data['model'].get('backend', config.model.backend),
                    magface=MagFaceConfig(
                        weights=magface_data.get('weights', config.model.magface.weights),
                        arch=magface_data.get('arch', config.model.magface.arch),
                        embedding_size=magface_data.get('embedding_size', config.model.magface.embedding_size)
                    )
                )
            
            if 'detection' in data:
                det_size = data['detection'].get('det_size', list(config.detection.det_size))
                config.detection = DetectionConfig(
                    det_size=tuple(det_size),
                    det_thresh=data['detection'].get('det_thresh', config.detection.det_thresh),
                    nms_thresh=data['detection'].get('nms_thresh', config.detection.nms_thresh),
                    max_num=data['detection'].get('max_num', config.detection.max_num)
                )
            
            if 'recognition' in data:
                config.recognition = RecognitionConfig(
                    select_largest=data['recognition'].get('select_largest', config.recognition.select_largest)
                )
            
            if 'matching' in data:
                config.matching = MatchingConfig(
                    similarity_threshold=data['matching'].get('similarity_threshold', config.matching.similarity_threshold)
                )
            
            if 'index' in data:
                config.index = IndexConfig(
                    url=data['index'].get('url', config.index.url),
                    api_key=data['index'].get('api_key', config.index.api_key),
                    collection_name=data['index'].get('collection_name', config.index.collection_name)
                )
            
            if 'paths' in data:
                config.paths = PathsConfig(
                    index_dir=data['paths'].get('index_dir', config.paths.index_dir),
                    gallery_dir=data['paths'].get('gallery_dir', config.paths.gallery_dir),
                    output_dir=data['paths'].get('output_dir', config.paths.output_dir)
                )
            
            if 'filter' in data:
                filter_data = data['filter']
                anime_data = filter_data.get('anime_classifier', {})
                config.filter = FilterConfig(
                    enabled=filter_data.get('enabled', config.filter.enabled),
                    min_det_score=filter_data.get('min_det_score', config.filter.min_det_score),
                    min_face_size=filter_data.get('min_face_size', config.filter.min_face_size),
                    min_quality_score=filter_data.get('min_quality_score', config.filter.min_quality_score),
                    anime_classifier=AnimeClassifierConfig(
                        enabled=anime_data.get('enabled', config.filter.anime_classifier.enabled),
                        threshold=anime_data.get('threshold', config.filter.anime_classifier.threshold)
                    )
                )
    
    return config


_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    global _config
    _config = load_config(config_path)
    return _config
