# MagFace-only simplified version
# Lazy imports to avoid requiring all dependencies for all use cases

__version__ = "0.1.0"


def get_config(*args, **kwargs):
    """Lazy import of config loader."""
    from .config_loader import get_config as _get_config
    return _get_config(*args, **kwargs)


def Config(*args, **kwargs):
    """Lazy import of Config class."""
    from .config_loader import Config as _Config
    return _Config(*args, **kwargs)


# FaceIndex requires qdrant_client, only import when needed
def FaceIndex(*args, **kwargs):
    """Lazy import of FaceIndex (requires qdrant_client)."""
    from .matching.face_index import FaceIndex as _FaceIndex
    return _FaceIndex(*args, **kwargs)


__all__ = [
    "FaceIndex",
    "get_config",
    "Config",
]
