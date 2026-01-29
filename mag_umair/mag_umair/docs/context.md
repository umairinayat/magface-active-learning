# Facial Matching System - Complete Project Context

> **Purpose:** This document serves as a comprehensive handover for LLMs or developers to understand and continue work on this project.

---

## 1. Project Overview

### What This Project Is

A **face recognition and clustering library** (`facelib`) that:
1. Detects faces in images using RetinaFace
2. Generates 512-dimensional embeddings using MagFace (iResNet100)
3. Stores embeddings in Qdrant vector database for similarity search
4. Automatically clusters faces belonging to the same person
5. Provides a web demo for testing and cluster management

### End Goal

Build a production-ready face matching system that can:
- Index large galleries of images (thousands of faces)
- Search for matching faces with high accuracy
- Automatically group faces of the same person into clusters
- Allow manual cluster management (merge, split, reassign)

### Current State

✅ **Fully Implemented:**
- Face detection (RetinaFace)
- Face embedding (MagFace iResNet100)
- Vector storage & search (Qdrant)
- Quality filtering (detection score, face size, anime detection)
- Clustering via connected components (NetworkX)
- Web demo with indexing, search, and cluster management UI

---

## 2. Project Structure

```
facial_matching_dan/
├── facelib/                    # Main library (public API)
│   ├── __init__.py             # Exports FaceLibrary, exceptions, types
│   ├── library.py              # FaceLibrary class (core implementation)
│   ├── exceptions.py           # Custom exceptions
│   └── types.py                # Dataclasses (SearchResult, FaceInfo, etc.)
│
├── src/                        # Internal components
│   ├── detection/              # RetinaFace detector
│   ├── embedding/              # MagFace embedder
│   ├── matching/               # FaceIndex (Qdrant wrapper)
│   ├── filter/                 # Quality filters, anime classifier
│   └── config_loader.py        # YAML config loading
│
├── demo/                       # Web demo application
│   ├── backend/
│   │   └── app_facelib.py      # FastAPI server
│   └── frontend/
│       ├── index.html          # Main search demo
│       └── clusters.html       # Cluster management UI
│
├── models/                     # Model weights (not in git)
│   └── magface_epoch_00025.pth # MagFace weights (~250MB)
│
├── config.yaml                 # Configuration file
└── data/                       # Index data, galleries
```

---

## 3. Core Components

### 3.1 FaceLibrary (`facelib/library.py`)

The main public API. Key methods:

| Method | Description |
|--------|-------------|
| `index(image, image_id)` | Index single image, auto-cluster faces |
| `index_batch(images)` | Index multiple images, cluster at end |
| `index_directory(path)` | Index all images in directory |
| `search(image, top_k, threshold)` | Find similar faces |
| `cluster_all(threshold)` | Re-cluster all faces |
| `get_cluster(cluster_id)` | Get all faces in a cluster |
| `merge_clusters(cluster_ids)` | Combine multiple clusters |
| `reassign_face(face_id, target_cluster_id)` | Move face to different cluster |
| `clear()` | Delete all indexed faces |

**Properties:**
- `embedder` — MagFaceEmbedder (lazy loaded)
- `detector` — RetinaFaceDetector (lazy loaded)
- `face_index` — FaceIndex (Qdrant wrapper)
- `size` — Number of indexed faces

### 3.2 FaceIndex (`src/matching/face_index.py`)

Qdrant wrapper with methods:
- `add(embedding, metadata)` — Add single face
- `search(embedding, top_k, threshold)` — Similarity search
- `scroll_all()` — Iterate all points (for clustering)
- `get_by_filter(field, value)` — Get points by metadata
- `update_payload(point_id, payload)` — Update metadata
- `delete_by_filter(field, value)` — Delete points
- `clear()` — Delete all points

### 3.3 Configuration (`config.yaml`)

```yaml
model:
  backend: "magface"
  magface:
    weights: "models/magface_epoch_00025.pth"
    arch: "iresnet100"
    embedding_size: 512

detection:
  det_thresh: 0.5

index:
  url: "http://localhost:6333"  # Qdrant URL
  collection_name: "face_embeddings"

filter:
  enabled: true
  min_det_score: 0.5
  min_face_size: 16
  min_quality_score: 22.0
  anime_classifier:
    enabled: true
    threshold: 0.6
```

---

## 4. Data Model

### Qdrant Payload Schema

Each face stored in Qdrant has this metadata:

```json
{
  "face_id": "photo123_face0",    // Unique ID: {image_id}_face{index}
  "image_id": "photo123",          // Source image identifier
  "bbox": [100, 50, 200, 180],     // Bounding box [x1, y1, x2, y2]
  "cluster_id": 42,                // Cluster assignment (null = unclustered)
  "cluster_confidence": 0.85,      // How confident the clustering is
  "det_score": 0.99,               // Detection confidence
  "embedding_norm": 28.5,          // MagFace quality (magnitude)
  "indexed_at": "2026-01-24T12:00:00Z"
}
```

### Key Types (`facelib/types.py`)

```python
@dataclass
class SearchResult:
    face_id: str
    image_id: str
    score: float                    # Cosine similarity
    bbox: Optional[Tuple[float, ...]]
    cluster_id: Optional[int]
    metadata: Dict[str, Any]
```

---

## 5. Clustering System

### Algorithm

Uses **NetworkX connected components**:
1. Query Qdrant for each face's neighbors (score ≥ threshold)
2. Build graph: face → neighbor edges
3. Find connected components = clusters
4. Assign auto-incremented cluster IDs

### Clustering Strategy

| Method | Behavior |
|--------|----------|
| `index(image)` | Index + immediate auto-clustering |
| `index_batch(images)` | Index all, then `cluster_all()` once |
| `index_directory(path)` | Same as batch |

**Why different strategies?**
- Single image: User expects immediate clustering
- Bulk import: Full clustering at end is more efficient and accurate

### Cluster State

- `cluster_id` stored in Qdrant payload (single source of truth)
- Next cluster ID persisted in `data/index/cluster_state.txt`
- Use `reassign_face(face_id, -1)` to create new cluster

---

## 6. Web Demo

### Starting the Demo

```bash
# Requires Qdrant running on localhost:6333
docker run -p 6333:6333 qdrant/qdrant

# Start FastAPI backend
cd demo/backend
python app_facelib.py
# → http://localhost:8000
```

### Demo Pages

| URL | Purpose |
|-----|---------|
| `/demo` | Main face search interface |
| `/clusters` | Cluster management (view, merge, split, reassign) |

### Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gallery/index/stream` | GET | Index directory with progress streaming |
| `/api/query` | POST | Search for face matches |
| `/api/detect_faces` | POST | Detect faces in image |
| `/api/clusters/list` | GET | List all clusters |
| `/api/clusters/{id}` | GET | Get cluster details |
| `/api/clusters/cluster-all` | POST | Re-run full clustering |
| `/api/clusters/merge` | POST | Merge clusters |
| `/api/clusters/reassign` | POST | Move face to different cluster |
| `/api/face/{face_id}` | GET | Get cropped face image |
| `/api/image/{image_id}` | GET | Get source image |

### Image Serving

The demo tracks `indexed_source_dir` to serve images:
1. Search in last indexed directory
2. Fall back to `demo/gallery/`
3. Fall back to `data/` directory (recursive)

---

## 7. Important Implementation Details

### Working Directory Issue

**Problem:** MagFace model paths are relative (`models/magface_epoch_00025.pth`).

**Solution:** The FastAPI app changes working directory to project root on startup:
```python
@app.on_event("startup")
async def startup():
    os.chdir(project_root)  # Ensures model paths resolve correctly
```

### Lazy Loading

Components are lazily initialized to speed up import:
- `FaceLibrary` uses `__getattr__` for lazy import
- `embedder`, `detector`, `face_index` are properties that init on first access

### Import Order

**CRITICAL:** PyTorch must be imported before TensorFlow to avoid segfault on macOS:
```python
import torch  # noqa: F401 - must be first
```

### No Local Registry

The local face registry (`_face_registry` dict, pickle save/load) has been **removed**. Qdrant is the single source of truth. No need to call `save()` — Qdrant auto-persists.

---

## 8. Exceptions

| Exception | When Raised |
|-----------|-------------|
| `NoFaceDetectedError` | No face found in image |
| `FilterRejectedError` | All faces rejected by quality filters |
| `FaceNotFoundError` | face_id doesn't exist in index |
| `ImageLoadError` | Cannot load/decode image file |

---

## 9. Current Caveats & Limitations

### Known Issues

1. **Anime/Cartoon Detection:** The anime classifier may have false positives on stylized photos. Threshold is configurable.

2. **Clustering Threshold:** Default 0.45 may need tuning per dataset. Lower = more aggressive grouping, higher = more conservative.

3. **Large Clusters:** Very large clusters (>1000 faces) may slow down `get_cluster()` queries.

4. **No Face Alignment:** Current pipeline uses simple bbox crop + resize. Proper 5-point alignment could improve accuracy.

5. **Model Weights:** The MagFace weights file (~250MB) is not in git. Must be downloaded separately.

### Demo-Specific

1. **Image Paths:** Images must be accessible from the server. The demo tracks `indexed_source_dir` but loses it on restart.

2. **Split UI:** The split cluster feature requires manual selection — there's no automatic "suggested splits" yet.

---

## 10. Dependencies

### Python Packages

```
torch>=2.0
onnxruntime
numpy
opencv-python
pillow
qdrant-client
networkx
fastapi
uvicorn
python-multipart
```

### External Services

- **Qdrant** — Vector database (run via Docker or cloud)

### Model Files

- `models/magface_epoch_00025.pth` — MagFace iResNet100 weights

---

## 11. Future Work / Open Items

- [ ] `find_cluster(image)` — Convenience method to identify who's in an image
- [ ] `list_clusters()` — List all clusters with face counts (currently done in demo backend)
- [ ] User-assigned cluster names/labels
- [ ] Cluster quality metrics and audit
- [ ] Automatic split suggestions based on embedding distances
- [ ] Face alignment with 5-point landmarks
- [ ] GPU acceleration for batch processing
- [ ] REST API authentication

---

## 12. Quick Reference

### Basic Usage

```python
from facelib import FaceLibrary

# Initialize
lib = FaceLibrary()

# Index images
lib.index("photo1.jpg")
lib.index_directory("/path/to/photos")

# Search
results = lib.search("query.jpg", top_k=10)
for r in results:
    print(f"{r.face_id}: {r.score:.2%} (cluster {r.cluster_id})")

# Cluster management
lib.merge_clusters([1, 2, 3])  # Merge into one
lib.reassign_face("photo1_face0", 5)  # Move to cluster 5
lib.reassign_face("photo2_face0", -1)  # Create new cluster
```

### Run Demo

```bash
# Terminal 1: Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2: Start demo
source .venv/bin/activate
cd demo/backend && python app_facelib.py

# Open browser: http://localhost:8000/demo
```

---

## 13. Contact / History

This project was developed incrementally with the following major phases:
1. **Phase 1:** Basic detection, embedding, and search
2. **Phase 2A:** Quality filtering, anime detection
3. **Phase 2B:** Clustering integration (current)

Last updated: January 2026
