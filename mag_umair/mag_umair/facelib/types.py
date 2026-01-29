# Type definitions for FaceLibrary

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class FaceInfo:
    """Information about a single detected face."""
    det_score: float
    embedding_norm: float # dont we also keep the embedding itself ?
    face_size: Tuple[int, int]
    bbox: Tuple[float, float, float, float]
    passed: bool = True
    reasons: List[str] = field(default_factory=list)


@dataclass
class FilterResult:
    """Result from filtering an image."""
    is_valid: bool
    reason: Optional[str]  # "cartoon", "no_face", "low_quality", or None
    faces: List[FaceInfo] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single search result."""
    face_id: str
    image_id: str
    score: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    cluster_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceRecord:
    """Complete record for an indexed face."""
    face_id: str
    image_id: str
    face_index: int
    embedding: np.ndarray
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[int] = None
