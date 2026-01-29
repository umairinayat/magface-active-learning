# MagFace-only simplified version
from .matching.face_index import FaceIndex
from .config_loader import get_config, Config

__version__ = "0.1.0"

__all__ = [
    "FaceIndex",
    "get_config",
    "Config",
]
