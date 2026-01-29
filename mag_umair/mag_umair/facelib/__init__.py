



# no have heavy dependencies
from .exceptions import (
    FaceLibraryError,
    NoFaceDetectedError,
    FilterRejectedError,
    FaceNotFoundError,
    IndexNotLoadedError,
    ImageLoadError,
)
from .types import (
    FilterResult,
    SearchResult,
    FaceRecord,
    FaceInfo,
)

__version__ = "1.0.0-alpha"

# Lazy import 
_FaceLibrary = None

def __getattr__(name):
    """Lazy loading for heavy imports."""
    global _FaceLibrary
    if name == "FaceLibrary":
        if _FaceLibrary is None:
            from .library import FaceLibrary as _FL
            _FaceLibrary = _FL
        return _FaceLibrary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main class
    "FaceLibrary",
    # Exceptions
    "FaceLibraryError",
    "NoFaceDetectedError",
    "FilterRejectedError",
    "FaceNotFoundError",
    "IndexNotLoadedError",
    "ImageLoadError",
    # Types
    "FilterResult",
    "SearchResult",
    "FaceRecord",
    "FaceInfo",
]

