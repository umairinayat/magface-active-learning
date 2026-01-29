# Clustering System Design: Schema & API

## 1. Terminology (Clarified)

| Term | Definition | Example |
|------|------------|---------|
| **face_id** | Unique ID for one detected face (one bbox from one image) | `"photo123_face0"` |
| **image_id** | Unique ID for the source image | `"photo123"` |
| **person_id** | Unique ID for a person (cluster of similar faces) | `"person_a1b2c3"` |

**Your understanding is correct:**
- `face_id` = one cropped bounding box = one face instance
- `person_id` = `cluster_id` = same thing (we'll use **`person_id`** going forward for clarity)

---

## 2. Current State (What We Store Now)

### Qdrant Payload (per face)
```json
{
  "face_id": "photo123_face0",
  "image_id": "photo123",
  "bbox": [100, 50, 200, 180]
}
```

### Local Registry (`_face_registry`) — FaceRecord
```python
FaceRecord(
    face_id="photo123_face0",
    image_id="photo123",
    face_index=0,           # Position in detection order
    embedding=np.array(...), # 512-dim vector (redundant with Qdrant)
    bbox=(100, 50, 200, 180),
    metadata={},             # User-provided metadata
    cluster_id=None          # Not used yet
)
```

### Problems with Current State
1. **Redundant storage** — embedding in both Qdrant and registry
2. **`cluster_id` is `None`** — not implemented
3. **Two sources of truth** — sync risk

---

## 3. Proposed Schema (After Clustering)

### Single Source of Truth: Qdrant

**Drop the local registry entirely.** Store everything in Qdrant payload.

### Qdrant Payload (per face)
```json
{
  "face_id": "photo123_face0",
  "image_id": "photo123",
  "bbox": [100, 50, 200, 180],
  "person_id": "person_a1b2c3",
  "person_confidence": 0.85,
  "det_score": 0.99,
  "quality_score": 28.5,
  "indexed_at": "2026-01-24T12:00:00Z"
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `face_id` | string | Primary key. Format: `{image_id}_face{n}` |
| `image_id` | string | Source image identifier |
| `bbox` | int[4] | Bounding box `[x1, y1, x2, y2]` in pixels |
| `person_id` | string | Cluster/person ID. `null` if not yet clustered |
| `person_confidence` | float | Confidence of person assignment (0.0-1.0) |
| `det_score` | float | Detection confidence from RetinaFace |
| `quality_score` | float | MagFace embedding magnitude (higher = better) |
| `indexed_at` | string | ISO timestamp of when face was indexed |

### Person Metadata (Optional Separate Collection)

For per-person aggregated info, consider a lightweight JSON file or separate Qdrant collection:

```json
{
  "person_a1b2c3": {
    "face_count": 47,
    "representative_face_id": "photo456_face0",
    "name": null,
    "created_at": "2026-01-24T12:00:00Z",
    "updated_at": "2026-01-24T12:00:00Z"
  }
}
```

This is optional — you can always compute face_count via Qdrant filter query.

---

## 4. API Design

### 4.1 Clustering Operations

#### `cluster_all() -> int`
Run full clustering on all indexed faces.

```python
def cluster_all(similarity_threshold: float = 0.6) -> int:
    """
    Cluster all faces and assign person_ids.
    
    Returns: Number of persons (clusters) created.
    """
```

**When to call:** Initial clustering, or periodic re-clustering.

---

#### `assign_person(face_id: str) -> str`
Assign a single new face to existing person or create new.

```python
def assign_person(face_id: str, threshold: float = 0.6) -> str:
    """
    Assign face to best-matching person or create new person.
    
    Returns: person_id (existing or newly created).
    """
```

**When to call:** After indexing a new face.

---

#### `get_person(person_id: str) -> dict`
Get all faces belonging to a person.

```python
def get_person(person_id: str) -> dict:
    """
    Returns:
    {
        "person_id": "person_a1b2c3",
        "face_count": 47,
        "faces": [
            {"face_id": "...", "image_id": "...", "bbox": [...], ...},
            ...
        ]
    }
    """
```

---

#### `merge_persons(person_ids: List[str]) -> str`
Merge multiple persons into one (same person, was incorrectly split).

```python
def merge_persons(person_ids: List[str], target_person_id: str = None) -> str:
    """
    Merge all faces from person_ids into one person.
    
    Args:
        person_ids: List of person IDs to merge.
        target_person_id: Optional. Use this as the merged ID. 
                          If None, uses first person_id.
    
    Returns: The resulting person_id.
    """
```

---

#### `split_person(person_id: str, face_groups: List[List[str]]) -> List[str]`
Split one person into multiple (different people, incorrectly merged).

```python
def split_person(person_id: str, face_groups: List[List[str]]) -> List[str]:
    """
    Split person into multiple persons.
    
    Args:
        person_id: Person to split.
        face_groups: List of face_id groups, each becomes a new person.
    
    Returns: List of new person_ids.
    """
```

---

#### `reassign_face(face_id: str, target_person_id: str) -> None`
Move one face to a different person.

```python
def reassign_face(face_id: str, target_person_id: str) -> None:
    """Move a single face to a different person."""
```

---

### 4.2 Query Operations

#### `search(image, ...) -> List[SearchResult]`
**Already exists.** Returns matches with `person_id` included.

```python
@dataclass
class SearchResult:
    face_id: str
    image_id: str
    score: float
    bbox: Tuple[int, int, int, int]
    person_id: Optional[str]  # NEW: included from Qdrant payload
    metadata: Dict[str, Any]
```

---

#### `find_person(image) -> str`
Convenience method: detect face, search, return person_id of best match.

```python
def find_person(image: ImageInput, threshold: float = 0.6) -> Optional[str]:
    """
    Find which person is in the image.
    
    Returns: person_id if confident match, None otherwise.
    """
```

---

#### `list_persons(limit: int = 100, offset: int = 0) -> List[dict]`
List all persons with summary info.

```python
def list_persons(limit: int = 100, offset: int = 0) -> List[dict]:
    """
    Returns:
    [
        {"person_id": "person_a1b2c3", "face_count": 47},
        {"person_id": "person_d4e5f6", "face_count": 12},
        ...
    ]
    """
```

---

## 5. Migration Plan

### Step 1: Update Qdrant Payload Schema
Add new fields when indexing:

```python
self.face_index.add(
    embedding,
    metadata={
        'face_id': face_id,
        'image_id': image_id,
        'bbox': face_bbox_list,
        'person_id': None,           # NEW
        'person_confidence': None,   # NEW
        'det_score': det_score,      # NEW
        'quality_score': quality,    # NEW
        'indexed_at': datetime.now().isoformat()  # NEW
    }
)
```

### Step 2: Implement `cluster_all()`
- Build graph from Qdrant neighbors
- Run NetworkX connected components
- Generate `person_id` for each cluster
- Update Qdrant payloads

### Step 3: Implement `assign_person()`
- Called after each new face is indexed
- Query neighbors, find best person match
- Update payload with `person_id`

### Step 4: Deprecate Local Registry
- Remove `_face_registry` dict
- Remove pickle save/load
- All data lives in Qdrant

---

## 6. Summary: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Storage** | Qdrant + local registry | Qdrant only |
| **person_id** | Not implemented | Stored in Qdrant payload |
| **Clustering** | Not implemented | NetworkX connected components |
| **Incremental** | N/A | `assign_person()` after each index |
| **Merge/Split** | Not implemented | API methods update Qdrant payloads |

---

## 7. Open Decisions

1. **person_id format:** 
   - Option A: `person_{uuid[:12]}` (e.g., `person_a1b2c3d4e5f6`)
   - Option B: Auto-increment integer (e.g., `1`, `2`, `3`)
   - Option C: User-assigned names (e.g., `john_doe`)

2. **Person metadata storage:**
   - Option A: Compute on-demand via Qdrant queries
   - Option B: Separate `persons.json` file
   - Option C: Separate Qdrant collection

3. **Confidence threshold for assignment:**
   - High threshold (0.7): Fewer errors, more "unknown" persons
   - Low threshold (0.5): More assignments, some errors

**Recommendation:** Start with Option A for person_id, Option A for metadata (compute on-demand), and threshold 0.6. Adjust based on results.
