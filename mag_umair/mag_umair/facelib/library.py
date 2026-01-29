# CRITICAL: Import PyTorch BEFORE TensorFlow to avoid segfault
# This is a known issue when mixing TF and PyTorch on macOS
import torch  # noqa: F401 - must be first

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import pickle
import numpy as np
from PIL import Image
import cv2
import uuid

# Add src to path to import existing components
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.matching.face_index import FaceIndex
from src.config_loader import load_config, Config

from .exceptions import (
    NoFaceDetectedError,
    FilterRejectedError,
    FaceNotFoundError,
    IndexNotLoadedError,
    ImageLoadError,
)
from .types import FilterResult, SearchResult, FaceRecord, FaceInfo


ImageInput = Union[str, Path, np.ndarray, Image.Image, bytes]


class FaceLibrary:
    """MagFace-based face recognition library """
    
    def __init__(self, config_path: Optional[str] = None, debug_save_faces: bool = False):
        self._config = load_config(config_path) 
        
        self._index_dir = self._config.paths.index_dir
        self._device = self._config.model.device
        
        self._filter_enabled = self._config.filter.enabled
        self._min_det_score = self._config.filter.min_det_score
        self._min_face_size = self._config.filter.min_face_size
        self._min_quality = self._config.filter.min_quality_score
        
        self._debug_save_faces = debug_save_faces
        if self._debug_save_faces:
            self._debug_dir = Path("debug_faces")
            self._debug_dir.mkdir(exist_ok=True)
            print(f"Debug mode enabled: saving aligned faces to {self._debug_dir}")
        
        # Anime classifier (lazy init)
        self._anime_classifier = None
        self._anime_classifier_enabled = self._config.filter.anime_classifier.enabled
        self._anime_threshold = self._config.filter.anime_classifier.threshold
        
        # components, lazy init
        self._embedder = None  # MagFaceEmbedder
        self._detector = None  # RetinaFaceDetector
        self._index: Optional[FaceIndex] = None
        
        # Face registry: maps face_id -> FaceRecord
        self._face_registry: Dict[str, FaceRecord] = {}
        # Image to faces mapping: maps image_id -> list of face_ids
        self._image_to_faces: Dict[str, List[str]] = {}
        
        # Auto-load registry if exists
        index_path = Path(self._index_dir)
        if index_path.exists() and (index_path / "facelib_registry.pkl").exists():
            self.load()

    # Properties to access internal components
    
    @property
    def embedder(self):
        """Access the MagFaceEmbedder instance."""
        if self._embedder is None:
            from src.embedding.magface_embedder import MagFaceEmbedder
            magface_cfg = getattr(self._config.model, 'magface', None)
            if magface_cfg is None:
                raise ValueError("MagFace config not found. Set model.magface in config.yaml")
            self._embedder = MagFaceEmbedder(
                model_path=magface_cfg.weights,
                arch=magface_cfg.arch,
                embedding_size=getattr(magface_cfg, 'embedding_size', 512),
                device=self._device
            )
        return self._embedder
    
    @property
    def detector(self):
        """Access the RetinaFace detector."""
        if self._detector is None:
            from src.detection.retinaface_detector import RetinaFaceDetector
            self._detector = RetinaFaceDetector(
                threshold=self._config.detection.det_thresh
            )
        return self._detector
    
    @property
    def face_index(self) -> FaceIndex:
        """Access the FaceIndex instance."""
        if self._index is None:
            self._index = FaceIndex()
        return self._index
    
    @property
    def clusterer(self):
        """Access the FaceClusterer instance (not implemented)."""
        raise NotImplementedError(
            "Clusterer will be implemented in Phase 2B: Clustering & ID Assignment"
        )
    
    @property
    def anime_classifier(self):
        """Access the AnimeRealClassifier instance (lazy init)."""
        if self._anime_classifier is None and self._anime_classifier_enabled:
            from src.filter.anime_real_classifier import AnimeRealClassifier
            self._anime_classifier = AnimeRealClassifier(
                model_name=self._config.filter.anime_classifier.model,
                threshold=self._anime_threshold
            )
        return self._anime_classifier
    
    def _get_faces_magface(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Internal: Detect and extract face embeddings using MagFace pipeline.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            List of face dicts with keys:
            - embedding: normalized 512-dim vector
            - bbox: [x1, y1, x2, y2]
            - det_score: detection confidence
            - landmarks: 5-point landmarks
            - quality_score: MagFace magnitude (higher = better quality)
        """
        
        # Check for anime/cartoon images first (if enabled)
        if self._filter_enabled and self._anime_classifier_enabled:
            try:
                result = self.anime_classifier.predict(image)
                if result['is_anime']:
                    return []  # Skip anime/cartoon images
            except Exception:
                pass  # If classifier fails, continue with detection
        
        # Detect faces with RetinaFace
        detections = self.detector.detect(image)
        if not detections:
            return []
        
        results = []
        for det in detections:
            # Crop face using bbox with margin
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            margin = 0.3
            
            x1_exp = max(0, int(x1 - w * margin))
            y1_exp = max(0, int(y1 - h * margin))
            x2_exp = min(image.shape[1], int(x2 + w * margin))
            y2_exp = min(image.shape[0], int(y2 + h * margin))
            
            face_crop = image[y1_exp:y2_exp, x1_exp:x2_exp]
            face_resized = cv2.resize(face_crop, (112, 112))
            
            if self._debug_save_faces:
                debug_id = str(uuid.uuid4())[:8]
                debug_path = self._debug_dir / f"index_{debug_id}.png"
                cv2.imwrite(str(debug_path), face_resized)
            
            # Get embedding and quality score from MagFace
            embedding, magnitude = self.embedder.get_embedding(face_resized)
            
            # Calculate face size
            bbox = det['bbox']
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_size = min(face_width, face_height)
            
            # Apply filters if enabled
            if self._filter_enabled:
                if det['det_score'] < self._min_det_score:
                    continue  # Skip low detection score
                if face_size < self._min_face_size:
                    continue  # Skip too small faces
                if magnitude < self._min_quality:
                    continue  # Skip low quality faces
            
            results.append({
                'embedding': embedding,
                'bbox': det['bbox'],
                'det_score': det['det_score'],
                'landmarks': det['landmarks'],
                'quality_score': magnitude,  # MagFace magnitude = quality
            })
        
        return results
    
    def detect_faces(self, image: ImageInput, auto_pad: bool = True) -> List[FaceInfo]:
        """
        Detect faces in image and return face info (detection only, no embedding).
        
        Args:
            image: Input image
            auto_pad: Add padding if no faces detected (helps with frame-filling faces)
            
        Returns:
            List of FaceInfo objects
        """
        # Handle bytes input
        if isinstance(image, bytes):
            image = self._bytes_to_image(image)
        
        # Load image as numpy array for processing
        img_array = self._load_image(image)
        
        # Use detector directly (no embedding computation)
        detections = self.detector.detect(img_array)
        
        # Retry with padding if no faces
        if not detections and auto_pad:
            padded = self._pad_image(img_array, pad_ratio=0.7)
            detections = self.detector.detect(padded)
            if not detections:
                padded = self._pad_image(img_array, pad_ratio=1.4)
                detections = self.detector.detect(padded)
        
        if not detections:
            return []
        
        result = []
        for det in detections:
            bbox = det.get('bbox', (0, 0, 0, 0))
            if isinstance(bbox, list):
                bbox = tuple(bbox)
            
            result.append(FaceInfo(
                det_score=det.get('det_score', 0.0),
                embedding_norm=0.0,  # No embedding computed
                face_size=(int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                bbox=bbox,
                passed=True,
                reasons=[]
            ))
        return result
    
    @property
    def size(self) -> int:
        """Number of faces in the index."""
        return self.face_index.size

    def index(
        self,
        image: ImageInput,
        image_id: Optional[str] = None, # todo: why do we provide the image_id ? should't we make the settings like these persistent througout the one instance of library app ? 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:

        # Handle bytes input
        if isinstance(image, bytes):
            image = self._bytes_to_image(image)
        
        # Generate image_id from path if not provided
        if image_id is None:
            if isinstance(image, (str, Path)):
                image_id = Path(image).stem
            else:
                import uuid
                image_id = str(uuid.uuid4())[:8]
        
        # Get faces using MagFace pipeline
        img_array = self._load_image(image)
        faces = self._get_faces_magface(img_array)
        if not faces:
            raise NoFaceDetectedError()
        passed_faces = [{'embedding': f['embedding'], 'det_score': f['det_score'], 
                        'bbox': f['bbox'], 'quality_score': f.get('quality_score', 0)} for f in faces]
        
        # Index each passed face
        face_ids = []
        for i, face in enumerate(passed_faces):
            face_id = f"{image_id}_face{i}"
            
            embedding = face.get('embedding')
            if embedding is None:
                continue
            
            # Add to Qdrant index with bbox (convert numpy types to native Python)
            face_bbox = face.get('bbox')
            if face_bbox is not None:
                face_bbox_list = [int(x) for x in face_bbox]
            else:
                face_bbox_list = None
            
            self.face_index.add(
                embedding,
                metadata={
                    'face_id': face_id, 
                    'image_id': image_id,
                    'bbox': face_bbox_list
                }
            )
            
            # Store in registry with bbox
            face_bbox = face.get('bbox')
            if isinstance(face_bbox, list):
                face_bbox = tuple(face_bbox)
            
            self._face_registry[face_id] = FaceRecord(
                face_id=face_id,
                image_id=image_id,
                face_index=i,
                embedding=embedding,
                bbox=face_bbox,
                metadata=metadata or {},
                cluster_id=None
            )
            
            # Update image mapping
            if image_id not in self._image_to_faces:
                self._image_to_faces[image_id] = []
            self._image_to_faces[image_id].append(face_id)
            
            face_ids.append(face_id)
        
        return face_ids
    
    # todo: return the list of lists instead of flattened list
    def index_batch(
        self,
        images: List[ImageInput],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[List[str]]:
        """
        Batch index multiple images.
        
        Args:
            images: List of image inputs (paths, arrays, etc.)
            on_progress: Callback(current, total) for progress updates.
            
        Returns:
            List of lists - each inner list contains face_ids for one image.
            Empty list for images that failed (no face, rejected, or error).
        """
        results: List[List[str]] = []
        total = len(images)
        
        for i, image in enumerate(images):
            try:
                face_ids = self.index(image)
                results.append(face_ids)
            except (NoFaceDetectedError, FilterRejectedError, ImageLoadError) as e:
                # Log but continue with other images
                print(f"Skipping image {i}: {e}")
                results.append([])  # Empty list for failed images
            
            if on_progress:
                on_progress(i + 1, total)
        
        return results

    def index_directory(
        self,
        path: str,
        recursive: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[List[str]]:

        dir_path = Path(path)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        
        # Find all image files
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if recursive:
            image_files = [
                f for f in dir_path.rglob('*')
                if f.suffix.lower() in extensions
            ]
        else:
            image_files = [
                f for f in dir_path.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]
        
        # Delegate to index_batch
        return self.index_batch(
            [str(f) for f in image_files],
            on_progress=on_progress
        )
    
    def _flatten_face_ids(self, nested: List[List[str]]) -> List[str]:
        """Flatten nested face_ids list. Utility for backward compatibility."""
        return [fid for sublist in nested for fid in sublist]

    def search(
        self,
        image: ImageInput,
        top_k: int = 0,
        threshold: float = 0.4,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> List[SearchResult]:


        if self.face_index.size == 0:
            raise IndexNotLoadedError("Index is empty. Index some images first.")
        
        # Handle bytes input
        if isinstance(image, bytes):
            image = self._bytes_to_image(image)
        
        # Load image
        img_array = self._load_image(image)
        
        # Step 1: Detect faces only (no embedding computation yet)
        detections = self.detector.detect(img_array)
        
        # Retry with padding if no faces detected
        if not detections:
            image_padded = self._pad_image(img_array, pad_ratio=0.7)
            detections = self.detector.detect(image_padded)
            img_array = image_padded  # Use padded image for cropping
            if not detections:
                image_padded = self._pad_image(img_array, pad_ratio=1.4)
                detections = self.detector.detect(image_padded)
                img_array = image_padded
        
        if not detections:
            raise NoFaceDetectedError("No face detected in query image even after auto-padding")
        
        # Step 2: Select face based on bbox (before computing any embeddings)
        if bbox is not None:
            query_cx = (bbox[0] + bbox[2]) / 2
            query_cy = (bbox[1] + bbox[3]) / 2
            
            def distance_to_bbox(det):
                det_bbox = det['bbox']
                f_cx = (det_bbox[0] + det_bbox[2]) / 2
                f_cy = (det_bbox[1] + det_bbox[3]) / 2
                return (f_cx - query_cx) ** 2 + (f_cy - query_cy) ** 2
            
            selected_det = min(detections, key=distance_to_bbox)
        else:
            # Select largest face
            selected_det = max(detections, key=lambda det: (
                (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
            ))
        
        # Step 3: Compute embedding ONLY for the selected face
        det_bbox = selected_det['bbox']
        x1, y1, x2, y2 = det_bbox
        w, h = x2 - x1, y2 - y1
        margin = 0.3
        
        x1_exp = max(0, int(x1 - w * margin))
        y1_exp = max(0, int(y1 - h * margin))
        x2_exp = min(img_array.shape[1], int(x2 + w * margin))
        y2_exp = min(img_array.shape[0], int(y2 + h * margin))
        
        face_crop = img_array[y1_exp:y2_exp, x1_exp:x2_exp]
        face_resized = cv2.resize(face_crop, (112, 112))
        
        if self._debug_save_faces:
            debug_id = str(uuid.uuid4())[:8]
            debug_path = self._debug_dir / f"search_{debug_id}.png"
            cv2.imwrite(str(debug_path), face_resized)
        
        query_embedding, _ = self.embedder.get_embedding(face_resized)
    
        # Determine search limit
        # If top_k=0, fetch all results above threshold (use large limit)
        # If top_k>0, use it as the limit
        search_limit = top_k if top_k > 0 else 10000  # Large number for "all results"
    
        # Search the index
        raw_results = self.face_index.search(query_embedding, k=search_limit, threshold=threshold)
        
        # Convert to SearchResult
        results = []
        for r in raw_results:
            meta = r.get('metadata', {})
            face_id = meta.get('face_id', '')
            image_id = meta.get('image_id', '')
            
            # Get bbox directly from Qdrant metadata (primary source)
            bbox_from_qdrant = meta.get('bbox')
            if bbox_from_qdrant and isinstance(bbox_from_qdrant, list):
                bbox_tuple = tuple(bbox_from_qdrant)
            else:
                bbox_tuple = None
            
            # Fallback to registry for other metadata
            record = self._face_registry.get(face_id)
            
            results.append(SearchResult(
                face_id=face_id,
                image_id=image_id,
                score=r.get('score', 0.0),
                bbox=bbox_tuple,
                cluster_id=record.cluster_id if record else None,
                metadata=record.metadata if record else {}
            ))
        
        return results
    
    def search_batch(
        self,
        images: List[ImageInput],
        top_k: int = 10
    ) -> List[List[SearchResult]]:
        """Batch search (not implemented yet)."""
        raise NotImplementedError("search_batch will be implemented in a future phase")
    
    def compare(self, image1: ImageInput, image2: ImageInput) -> float:
        """Direct 1:1 similarity between two images (not implemented yet)."""
        raise NotImplementedError("compare will be implemented in a future phase")
    
    # clurstering methods ( stub yet )
    
    def cluster(self, algorithm: str = "chinese_whispers") -> int:
        """Cluster all indexed faces by person (not implemented)."""
        raise NotImplementedError(
            "Clustering will be implemented in Phase 2B: Clustering & ID Assignment"
        )
    
    def get_cluster(self, face_id: str) -> int:
        """Get cluster_id for a face (not implemented)."""
        raise NotImplementedError(
            "Clustering will be implemented in Phase 2B: Clustering & ID Assignment"
        )
    
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Get all face_ids in a cluster (not implemented)."""
        raise NotImplementedError(
            "Clustering will be implemented in Phase 2B: Clustering & ID Assignment"
        )
    
    def recluster(self) -> int:
        """Force full re-clustering (not implemented)."""
        raise NotImplementedError(
            "Clustering will be implemented in Phase 2B: Clustering & ID Assignment"
        )
    

    # utility methods

    
    def get_embedding(self, image: ImageInput) -> List[Tuple[np.ndarray, float]]:
        """
        Extract face embeddings from image without indexing.
        
        Args:
            image: Input image (path, PIL, numpy array, or bytes)
            
        Returns:
            List of (embedding, quality_score) tuples for each detected face.
            Empty list if no faces detected.
        """
        if isinstance(image, bytes):
            image = self._bytes_to_image(image)
        
        img_array = self._load_image(image)
        faces = self._get_faces_magface(img_array)
        
        return [(f['embedding'], f['quality_score']) for f in faces]
    
    def get_face(self, face_id: str) -> FaceRecord:
        if face_id not in self._face_registry:
            raise FaceNotFoundError(face_id)
        return self._face_registry[face_id]
    

    # Index Management

    def save(self) -> None:
        """Persist registry to disk. Qdrant index is auto-persisted."""
        index_path = Path(self._index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save face registry and mappings (Qdrant handles vector persistence)
        registry_data = {
            'face_registry': {
                fid: {
                    'face_id': r.face_id,
                    'image_id': r.image_id,
                    'face_index': r.face_index,
                    'embedding': r.embedding,
                    'bbox': r.bbox,  # Include bbox!
                    'metadata': r.metadata,
                    'cluster_id': r.cluster_id
                }
                for fid, r in self._face_registry.items()
            },
            'image_to_faces': self._image_to_faces
        }
        
        with open(index_path / "facelib_registry.pkl", 'wb') as f:
            pickle.dump(registry_data, f)
        
        print(f"FaceLibrary saved to {index_path}")
    
    def load(self) -> None:
        """Load registry from disk. Qdrant index is always available."""
        index_path = Path(self._index_dir)
        
        if not index_path.exists():
            raise IndexNotLoadedError(f"Index directory not found: {index_path}")
        
        # Load face registry (Qdrant handles vector persistence)
        registry_file = index_path / "facelib_registry.pkl"
        if registry_file.exists():
            with open(registry_file, 'rb') as f:
                data = pickle.load(f)
            
            self._face_registry = {
                fid: FaceRecord(
                    face_id=r['face_id'],
                    image_id=r['image_id'],
                    face_index=r['face_index'],
                    embedding=r['embedding'],
                    bbox=r.get('bbox'),  # Include bbox!
                    metadata=r['metadata'],
                    cluster_id=r['cluster_id']
                )
                for fid, r in data.get('face_registry', {}).items()
            }
            self._image_to_faces = data.get('image_to_faces', {})
        
        print(f"FaceLibrary loaded from {index_path}")
    
    def clear(self) -> None:
        self.face_index.clear()
        self._face_registry = {}
        self._image_to_faces = {}
    
    # todo: review this again as we implement clustering
    def remove(self, face_id: str) -> None:
        """Remove a face from the index."""
        if face_id not in self._face_registry:
            raise FaceNotFoundError(face_id)
        
        # Delete from Qdrant
        self.face_index.delete_by_filter('face_id', face_id)
        
        record = self._face_registry.pop(face_id)
        
        # Remove from image mapping
        if record.image_id in self._image_to_faces:
            self._image_to_faces[record.image_id] = [
                fid for fid in self._image_to_faces[record.image_id]
                if fid != face_id
            ]
            if not self._image_to_faces[record.image_id]:
                del self._image_to_faces[record.image_id]
    
    def remove_image(self, image_id: str) -> None:
        """Remove all faces from an image."""
        if image_id not in self._image_to_faces:
            return
        
        face_ids = self._image_to_faces[image_id].copy()
        for face_id in face_ids:
            if face_id in self._face_registry:
                del self._face_registry[face_id]
        
        del self._image_to_faces[image_id]
    
    def count(self) -> int:
        """Number of indexed faces."""
        return len(self._face_registry)
    

    # Private helpers
    
    def _bytes_to_image(self, data: bytes) -> Image.Image:
        """Convert bytes to PIL Image."""
        import io
        try:
            return Image.open(io.BytesIO(data))
        except Exception as e:
            raise ImageLoadError("bytes input", str(e))
    
    def _load_image(self, image: ImageInput) -> np.ndarray:
        """
        Load image as numpy array (BGR format for OpenCV/MagFace).
        
        Args:
            image: Path, PIL Image, numpy array, or bytes
            
        Returns:
            BGR numpy array
        """
        import cv2
        
        if isinstance(image, bytes):
            # Convert bytes to PIL then to numpy
            pil_img = self._bytes_to_image(image)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ImageLoadError(str(image), "Could not read image file")
            return img
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Assume already BGR
            return image
        else:
            raise ImageLoadError(str(type(image)), f"Unsupported image type: {type(image)}")
    
    def _pad_image(
        self,
        image: ImageInput,
        pad_ratio: float = 0.7,
        pad_color: tuple = (0, 0, 0)
    ) -> np.ndarray:
        """
        Add padding around image to help detect faces that fill the frame.
        
        Args:
            image: Input image.
            pad_ratio: Padding as fraction of image size (e.g., 0.7 = 70% padding).
            pad_color: BGR color for padding (default: black).
            
        Returns:
            Padded image as numpy array (BGR).
        """
        import cv2
        
        img = self._load_image(image)
        
        h, w = img.shape[:2]
        pad_h = int(h * pad_ratio)
        pad_w = int(w * pad_ratio)
        
        padded = cv2.copyMakeBorder(
            img,
            top=pad_h,
            bottom=pad_h,
            left=pad_w,
            right=pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )
        
        return padded

