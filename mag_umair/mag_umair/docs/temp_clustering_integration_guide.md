# Clustering Integration Guide

Where and how to modify the existing `facelib/library.py`.

---

## File: `facelib/library.py`

### 1. REMOVE: Local Registry (Lines 65-73)

**Current:**
```python
# Face registry: maps face_id -> FaceRecord
self._face_registry: Dict[str, FaceRecord] = {}
# Image to faces mapping: maps image_id -> list of face_ids
self._image_to_faces: Dict[str, List[str]] = {}

# Auto-load registry if exists
index_path = Path(self._index_dir)
if index_path.exists() and (index_path / "facelib_registry.pkl").exists():
    self.load()
```

**After:** Delete entirely.

---

### 2. ADD: Cluster State (After line ~63)

**Add:**
```python
# Clustering config
self._cluster_threshold = 0.45
self._next_cluster_id = 1  # Auto-increment counter
```

---

### 3. MODIFY: `clusterer` Property (Lines 111-115)

**Current:**
```python
@property
def clusterer(self):
    """Access the FaceClusterer instance (not implemented)."""
    raise NotImplementedError(...)
```

**After:** Remove this property entirely. We won't have a separate clusterer class.

---

### 4. MODIFY: `index()` Method (Lines 260-334)

**Location:** After storing face in Qdrant (around line 310)

**Current:**
```python
self.face_index.add(
    embedding,
    metadata={
        'face_id': face_id, 
        'image_id': image_id,
        'bbox': face_bbox_list
    }
)

# Store in registry with bbox
...
self._face_registry[face_id] = FaceRecord(...)
```

**After:**
```python
self.face_index.add(
    embedding,
    metadata={
        'face_id': face_id, 
        'image_id': image_id,
        'bbox': face_bbox_list,
        'cluster_id': None,              # NEW
        'cluster_confidence': None,       # NEW
        'det_score': float(face.get('det_score', 0)),  # NEW
        'embedding_norm': float(face.get('quality_score', 0)),  # NEW (renamed)
        'indexed_at': datetime.now().isoformat()  # NEW
    }
)

# NEW: Auto-assign to cluster (for single image indexing)
self._add_face_to_cluster(face_id, embedding)

# REMOVE: All registry code below
```

---

### 5. MODIFY: `index_batch()` Method (Lines 337-368)

**Current:** Calls `self.index()` for each image (which now does clustering).

**After:** Index all without clustering, then cluster at end.

```python
def index_batch(
    self,
    images: List[ImageInput],
    on_progress: Optional[Callable[[int, int], None]] = None
) -> List[List[str]]:
    results: List[List[str]] = []
    total = len(images)
    
    for i, image in enumerate(images):
        try:
            # NEW: Pass cluster=False to skip auto-clustering
            face_ids = self._index_single(image, cluster=False)
            results.append(face_ids)
        except (NoFaceDetectedError, FilterRejectedError, ImageLoadError) as e:
            print(f"Skipping image {i}: {e}")
            results.append([])
        
        if on_progress:
            on_progress(i + 1, total)
    
    # NEW: Cluster all faces at the end
    self.cluster_all()
    
    return results
```

**Note:** Refactor `index()` to `_index_single(image, cluster=True)` internally.

---

### 6. MODIFY: `index_directory()` Method (Lines 370-398)

**No change needed** — it delegates to `index_batch()` which now handles clustering.

---

### 7. MODIFY: `search()` Method (Lines 404-506)

**Location:** Where SearchResult is constructed (around line 497-504)

**Current:**
```python
results.append(SearchResult(
    face_id=face_id,
    image_id=image_id,
    score=r.get('score', 0.0),
    bbox=bbox_tuple,
    cluster_id=record.cluster_id if record else None,  # From registry
    metadata=record.metadata if record else {}
))
```

**After:**
```python
results.append(SearchResult(
    face_id=face_id,
    image_id=image_id,
    score=r.get('score', 0.0),
    bbox=bbox_tuple,
    cluster_id=meta.get('cluster_id'),  # NEW: From Qdrant payload
    metadata={}  # Or fetch from Qdrant if needed
))
```

---

### 8. ADD: New Clustering Methods (After `search()`, around line 507)

```python
# ============================================================
# CLUSTERING METHODS
# ============================================================

def cluster_all(self, threshold: float = None) -> int:
    """
    Run full clustering on all indexed faces.
    
    Returns: Number of clusters created.
    """
    if threshold is None:
        threshold = self._cluster_threshold
    
    # Step 1: Get all face_ids from Qdrant
    # Step 2: Build graph using Qdrant neighbor queries
    # Step 3: Run NetworkX connected_components
    # Step 4: Update Qdrant payloads with cluster_id
    # Step 5: Return cluster count
    pass

def _add_face_to_cluster(self, face_id: str, embedding: np.ndarray) -> int:
    """
    Private: Assign single face to cluster (called after indexing).
    
    Returns: Assigned cluster_id.
    """
    # Step 1: Query top-k neighbors
    # Step 2: Find best neighbor with cluster_id
    # Step 3: If similarity > threshold, join that cluster
    # Step 4: Else create new cluster (auto-increment)
    # Step 5: Update Qdrant payload
    pass

def get_cluster(self, cluster_id: int) -> dict:
    """
    Get all faces in a cluster.
    
    Returns: {"cluster_id": int, "face_count": int, "faces": [...]}
    """
    # Query Qdrant with filter: cluster_id == cluster_id
    pass

def merge_clusters(self, cluster_ids: List[int]) -> int:
    """
    Merge multiple clusters into one.
    
    Returns: Resulting cluster_id.
    """
    # Update all faces with cluster_id in cluster_ids to target_id
    pass

def split_cluster(self, cluster_id: int, face_groups: List[List[str]]) -> List[int]:
    """
    Split cluster into multiple clusters.
    
    Returns: List of new cluster_ids.
    """
    # Assign new cluster_ids to each face group
    pass

def reassign_face(self, face_id: str, target_cluster_id: int) -> None:
    """Move single face to different cluster."""
    # Update Qdrant payload for face_id
    pass

def _get_next_cluster_id(self) -> int:
    """Get next available cluster_id (auto-increment)."""
    cluster_id = self._next_cluster_id
    self._next_cluster_id += 1
    return cluster_id
```

---

### 9. REMOVE: Registry Methods (Lines 569-672)

**Remove entirely:**
- `get_face()` — uses `_face_registry`
- `save()` — saves pickle
- `load()` — loads pickle
- `clear()` — clears registry (keep Qdrant clear)
- `remove()` — uses `_face_registry`
- `remove_image()` — uses `_image_to_faces`
- `count()` — uses `_face_registry`

**Replace with Qdrant-based versions:**

```python
def get_face(self, face_id: str) -> dict:
    """Get face metadata from Qdrant."""
    results = self.face_index.get_by_filter('face_id', face_id)
    if not results:
        raise FaceNotFoundError(face_id)
    return results[0]

def clear(self) -> None:
    """Clear all indexed faces."""
    self.face_index.clear()
    self._next_cluster_id = 1

def remove(self, face_id: str) -> None:
    """Remove a face from the index."""
    self.face_index.delete_by_filter('face_id', face_id)

def remove_image(self, image_id: str) -> None:
    """Remove all faces from an image."""
    self.face_index.delete_by_filter('image_id', image_id)

def count(self) -> int:
    """Number of indexed faces."""
    return self.face_index.size
```

---

### 10. ADD: Import (Top of file)

```python
from datetime import datetime
import networkx as nx  # For clustering
```

---

## File: `facelib/types.py`

### Update `SearchResult` (Lines 27-35)

**Current:**
```python
@dataclass
class SearchResult:
    face_id: str
    image_id: str
    score: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    cluster_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**After:** No change needed — `cluster_id` already exists.

---

## File: `src/matching/face_index.py`

### May need to add methods:

```python
def get_by_filter(self, field: str, value: Any) -> List[dict]:
    """Get points matching filter."""
    # Qdrant scroll with filter
    pass

def update_payload(self, face_id: str, payload: dict) -> None:
    """Update payload for a point."""
    # Qdrant set_payload
    pass

def scroll_all(self, batch_size: int = 100) -> Iterator[dict]:
    """Iterate all points."""
    # Qdrant scroll
    pass
```

---

## Summary: Change Locations

| File | Lines | Change |
|------|-------|--------|
| `library.py` | 65-73 | Remove registry init |
| `library.py` | ~63 | Add cluster config |
| `library.py` | 111-115 | Remove `clusterer` property |
| `library.py` | 303-324 | Update `index()` payload, add clustering call |
| `library.py` | 337-368 | Update `index_batch()` to cluster at end |
| `library.py` | 497-504 | Update `search()` to get cluster_id from Qdrant |
| `library.py` | ~507 | Add new clustering methods |
| `library.py` | 569-672 | Replace registry methods with Qdrant-based |
| `library.py` | top | Add imports |
| `face_index.py` | — | Add helper methods if needed |

---

## Execution Order

1. Add imports and cluster config
2. Add new clustering methods (stubs first)
3. Update `index()` payload schema
4. Update `index_batch()` to cluster at end
5. Update `search()` to use Qdrant payload
6. Replace registry methods with Qdrant-based
7. Remove registry init and pickle code
8. Test single-image indexing
9. Test batch indexing
10. Test clustering methods
