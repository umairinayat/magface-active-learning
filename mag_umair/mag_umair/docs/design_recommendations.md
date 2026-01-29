# Design Recommendations

## Issue 1: Face Index Non-Determinism

### Problem

The frontend sends `face_index` (integer) to select which face to query. The backend re-runs detection and uses that index.

**Current flow:**

1. `/api/detect` → returns faces sorted by `det_score` descending
2. User clicks face at index `N`
3. `/api/query` with `face_index=N` → backend re-detects and picks `detections[N]`

**Risk:** If detection order changes between calls (due to floating-point instability in scores, or any preprocessing difference), wrong face is selected.

**In practice:** Within the same session/image, order is *likely* stable. But not guaranteed.

### Recommendation: Use Normalized Center Coordinates

Instead of `face_index`, send the **center position as percentage of image dimensions**:

```
x_pct = (x1 + x2) / 2 / image_width
y_pct = (y1 + y2) / 2 / image_height
```

**Benefits:**

- Detection-order independent
- Robust to minor bounding box fluctuations
- Backend matches by "closest center to (x_pct, y_pct)"

### Proposed Change

**Frontend:** Send `face_x_pct` and `face_y_pct` instead of `face_index`

**Backend `/api/query`:**

```python
if face_x_pct is not None and face_y_pct is not None:
    # Get image dimensions
    img = cv2.imread(query_path)
    h, w = img.shape[:2]
    target_x = face_x_pct * w
    target_y = face_y_pct * h
    # Select closest detection by center distance
    selected = min(detections, key=lambda d: dist_to_center(d, target_x, target_y))
```

**Effort:** Small (~20 lines frontend + backend)

---

## Issue 2: Local Registry Redundancy

### Problem

Two sources of truth exist:

1. **Qdrant** — stores embeddings + payload (`face_id`, `image_id`, `bbox`)
2. **Local registry** (`_face_registry`) — stores `FaceRecord` with same data plus `cluster_id`, `metadata`

This creates:

- Sync risk (data mismatch)
- Extra complexity
- Redundant storage

### Current Registry Usage

| Field          | In Qdrant?  | Used For                     |
| -------------- | ----------- | ---------------------------- |
| `face_id`    | ✅          | Primary key                  |
| `image_id`   | ✅          | Grouping                     |
| `bbox`       | ✅          | Display                      |
| `embedding`  | ✅ (vector) | Search                       |
| `cluster_id` | ❌          | Clustering (not implemented) |
| `metadata`   | ❌          | User-provided metadata       |
| `face_index` | ❌          | Original index in image      |

### Recommendation: Migrate Everything to Qdrant

**Phase 1 (now):** Add `metadata` dict to Qdrant payload

```python
payload = {
    'face_id': face_id,
    'image_id': image_id,
    'bbox': bbox_list,
    'metadata': user_metadata  # NEW
}
```

**Phase 2 (when clustering implemented):** Add `cluster_id` to Qdrant payload

**Phase 3:** Remove `_face_registry` entirely

### Benefits

- Single source of truth
- No sync issues
- Survives process restart without pickle loading
- Qdrant handles persistence

### Trade-offs

- Slightly higher latency for metadata lookups (network vs memory)
- Need to re-index to migrate existing data

### Effort

- Phase 1: ~15 lines
- Phase 2: ~10 lines
- Phase 3: ~50 lines (remove registry code, update all references)

---

## Summary

| Issue                      | Severity   | Recommended Action                                     |
| -------------------------- | ---------- | ------------------------------------------------------ |
| Face index non-determinism | Medium     | Replace `face_index` with `face_x_pct, face_y_pct` |
| Registry redundancy        | Low-Medium | Migrate all metadata to Qdrant, remove registry        |

Both changes improve robustness and reduce complexity. Neither is urgent for current functionality.
