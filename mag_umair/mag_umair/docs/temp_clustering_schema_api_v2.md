# Clustering System Design: Schema & API (v2)

## 1. Terminology

| Term | Definition | Example |
|------|------------|---------|
| **face_id** | Unique ID for one detected face (one bbox from one image) | `"photo123_face0"` |
| **image_id** | Unique ID for the source image | `"photo123"` |
| **cluster_id** | Unique ID for a cluster (group of similar faces = one person) | `42` |

---

## 2. Current State → What Changes

### Removing
- ❌ Local registry (`_face_registry` pickle file)
- ❌ `_image_to_faces` mapping

### Keeping (Single Source of Truth)
- ✅ Qdrant — stores embeddings + all metadata

---

## 3. Qdrant Payload Schema

### Per-Face Payload
```json
{
  "face_id": "photo123_face0",
  "image_id": "photo123",
  "bbox": [100, 50, 200, 180],
  "cluster_id": 42,
  "cluster_confidence": 0.85,
  "det_score": 0.99,
  "embedding_norm": 28.5,
  "indexed_at": "2026-01-24T12:00:00Z"
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `face_id` | string | Primary key. Format: `{image_id}_face{n}` |
| `image_id` | string | Source image identifier |
| `bbox` | int[4] | Bounding box `[x1, y1, x2, y2]` in pixels |
| `cluster_id` | int | Auto-incremented cluster ID. `null` if not yet clustered |
| `cluster_confidence` | float | Confidence of cluster assignment (0.0-1.0). `null` if not clustered |
| `det_score` | float | Detection confidence from RetinaFace |
| `embedding_norm` | float | MagFace embedding magnitude (higher = better quality) |
| `indexed_at` | string | ISO timestamp |

---

## 4. API Design

### 4.1 Indexing (Existing, Modified)

#### `index(image, image_id, metadata) -> List[str]`

**Change:** After indexing, automatically assigns `cluster_id` to each new face.

```python
def index(self, image, image_id=None, metadata=None) -> List[str]:
    """
    Index image: detect faces, compute embeddings, store in Qdrant,
    and automatically assign cluster_id.
    
    Returns: List of face_ids created.
    """
    # ... existing detection/embedding logic ...
    
    for face in faces:
        face_id = self._store_face_in_qdrant(face, image_id)
        self._add_face_to_cluster(face_id)  # NEW: auto-assign cluster
    
    return face_ids
```

**`_add_face_to_cluster(face_id)`** is private — called automatically.

---

### 4.2 Clustering Operations

#### `cluster_all(threshold: float = 0.45) -> int`
Run full clustering on all indexed faces.

```python
def cluster_all(self, threshold: float = 0.45) -> int:
    """
    Cluster all faces and assign cluster_ids.
    
    Args:
        threshold: Similarity threshold for connecting faces (default 0.45)
    
    Returns: Number of clusters created.
    """
```

**When to call:** 
- Initial clustering after bulk indexing
- Manual trigger if needed

---

#### `get_cluster(cluster_id: int) -> dict`
Get all faces in a cluster.

```python
def get_cluster(self, cluster_id: int) -> dict:
    """
    Returns:
    {
        "cluster_id": 42,
        "face_count": 47,
        "faces": [
            {"face_id": "...", "image_id": "...", "bbox": [...], ...},
            ...
        ]
    }
    """
```

---

#### `merge_clusters(cluster_ids: List[int]) -> int`
Merge multiple clusters into one.

```python
def merge_clusters(self, cluster_ids: List[int]) -> int:
    """
    Merge all faces from cluster_ids into one cluster.
    
    Returns: The resulting cluster_id (smallest of the input IDs).
    """
```

---

#### `split_cluster(cluster_id: int, face_groups: List[List[str]]) -> List[int]`
Split one cluster into multiple.

```python
def split_cluster(self, cluster_id: int, face_groups: List[List[str]]) -> List[int]:
    """
    Split cluster into multiple clusters.
    
    Args:
        cluster_id: Cluster to split.
        face_groups: List of face_id groups, each becomes a new cluster.
    
    Returns: List of new cluster_ids.
    """
```

---

#### `reassign_face(face_id: str, target_cluster_id: int) -> None`
Move one face to a different cluster.

```python
def reassign_face(self, face_id: str, target_cluster_id: int) -> None:
    """Move a single face to a different cluster."""
```

---

### 4.3 Search (Existing, Returns cluster_id)

#### `search(image, ...) -> List[SearchResult]`

```python
@dataclass
class SearchResult:
    face_id: str
    image_id: str
    score: float
    bbox: Tuple[int, int, int, int]
    cluster_id: Optional[int]  # Included from Qdrant payload
    metadata: Dict[str, Any]
```

---

## 5. Private Method: `_add_face_to_cluster(face_id)`

Called automatically after each face is indexed.

```python
def _add_face_to_cluster(self, face_id: str, threshold: float = 0.45) -> int:
    """
    Assign face to best-matching cluster or create new cluster.
    
    Logic:
    1. Query top-k neighbors from Qdrant
    2. Find neighbors with cluster_id assigned
    3. If best neighbor similarity > threshold:
       - Assign same cluster_id
       - Set cluster_confidence = similarity score
    4. Else:
       - Create new cluster_id (auto-increment)
       - Set cluster_confidence = 1.0
    
    Returns: Assigned cluster_id
    """
```

---

## 6. About Periodic Re-clustering

### Do we need it?

**Short answer: Probably not.**

**Why periodic re-clustering was considered:**
- Over time, incremental assignments may drift (chain effect)
- New faces might connect previously separate clusters

**Why you likely don't need it:**
- With threshold 0.45 and good embeddings, drift is minimal
- `merge_clusters()` handles manual corrections
- Full re-clustering is expensive at 2M scale

**Recommendation:** 
- Don't implement periodic re-clustering now
- If you see quality issues later, you can always call `cluster_all()` manually
- Or implement a "quality audit" that flags suspicious clusters for review

---

## 7. Cluster Metadata Storage — Fork in the Road

### The Question
Where to store aggregate cluster info like: `face_count`, `name`, `representative_face_id`?

### Option A: Compute On-Demand (Recommended for now)

**How:** Query Qdrant when you need it.

```python
def get_cluster(self, cluster_id: int) -> dict:
    # Query Qdrant: filter by cluster_id
    faces = qdrant.scroll(filter={"cluster_id": cluster_id})
    return {
        "cluster_id": cluster_id,
        "face_count": len(faces),
        "faces": faces
    }
```

**Pros:**
- No extra storage
- Always accurate (no sync issues)
- Minimal implementation

**Cons:**
- Slower for listing all clusters with counts
- Can't store user-assigned names without somewhere to put them

### Option B: Separate JSON File

**How:** `data/index/clusters.json`

```json
{
  "42": {"face_count": 47, "name": "John Doe"},
  "43": {"face_count": 12, "name": null}
}
```

**Pros:**
- Fast lookups
- Can store user-assigned names

**Cons:**
- Must keep in sync with Qdrant
- Another file to manage

### Option C: Separate Qdrant Collection

**How:** Collection `cluster_metadata` with one entry per cluster.

**Pros:**
- All data in Qdrant
- Queryable

**Cons:**
- More complex
- Overkill for simple metadata

### Recommendation

**Start with Option A (compute on-demand).**

You said you want to keep things minimal and don't know the client's system yet. Option A:
- Zero extra storage
- Works now
- Easy to switch to Option B later if needed

If the client needs user-assigned cluster names, add Option B (JSON file) at that point.

---

## 8. cluster_id Generation

**Auto-incremented integer.**

```python
def _get_next_cluster_id(self) -> int:
    """Get next available cluster_id."""
    # Query Qdrant for max cluster_id
    result = qdrant.scroll(
        limit=1,
        order_by={"key": "cluster_id", "direction": "desc"}
    )
    if result:
        return result[0].payload["cluster_id"] + 1
    return 1
```

Or simpler: maintain a counter in a small state file.

---

## 9. Summary: What We're Building

### Methods to Implement

| Method | Public/Private | Description |
|--------|----------------|-------------|
| `cluster_all()` | Public | Full clustering of all faces |
| `_add_face_to_cluster()` | Private | Auto-called after indexing |
| `get_cluster()` | Public | Get faces in a cluster |
| `merge_clusters()` | Public | Combine clusters |
| `split_cluster()` | Public | Divide a cluster |
| `reassign_face()` | Public | Move one face |

### NOT Implementing Yet
- ❌ Periodic re-clustering
- ❌ Cluster metadata storage (Option B/C)
- ❌ `find_cluster()` convenience method
- ❌ `list_clusters()` method

### Key Parameters
- **Default threshold:** `0.45`
- **cluster_id format:** Auto-incremented integer
- **Storage:** Qdrant only (no local registry)

---

## 10. Migration Checklist

- [ ] Remove `_face_registry` dict from `FaceLibrary`
- [ ] Remove pickle save/load methods
- [ ] Update `index()` to store new payload fields
- [ ] Update `index()` to call `_add_face_to_cluster()`
- [ ] Implement `cluster_all()`
- [ ] Implement `_add_face_to_cluster()`
- [ ] Implement `get_cluster()`
- [ ] Implement `merge_clusters()`
- [ ] Implement `split_cluster()`
- [ ] Implement `reassign_face()`
- [ ] Update `SearchResult` to include `cluster_id`
- [ ] Update `search()` to return `cluster_id` from payload
