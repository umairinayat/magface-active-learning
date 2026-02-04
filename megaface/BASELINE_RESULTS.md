# MagFace Baseline Results

**Date**: 2026-02-04  
**Dataset**: MegaFace (100,000 images)  
**Model**: magface_epoch_00025.pth (iresnet100, 512-dim embeddings)

---

## Dataset Statistics

| Subset | Images | Identities |
|--------|--------|------------|
| FaceScrub (probe) | 3,530 | 80 |
| MegaFace (distractors) | 96,470 | - |
| **Total** | **100,000** | - |

---

## 1. Embedding-Based Verification Metrics

Using cosine similarity between 512-dim embeddings.

### ROC & Accuracy

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.9953 |
| **Best Accuracy** | 99.58% |
| **Best Threshold** | 0.3076 |

### TAR @ FAR (True Accept Rate @ False Accept Rate)

| FAR | TAR |
|-----|-----|
| 1e-4 | 97.98% |
| 1e-3 | 98.31% |
| 1e-2 | 98.53% |
| 1e-1 | 99.06% |

### Score Distribution

| Metric | Positive Pairs | Negative Pairs |
|--------|---------------|----------------|
| Mean | 0.6955 | 0.0216 |
| Std | 0.1308 | 0.0749 |

### Identification

| Metric | Value |
|--------|-------|
| **Rank-1 Accuracy** | 99.29% |
| Correct / Total | 3,505 / 3,530 |

---

## 2. Pair Verification (visualize_errors.py)

Threshold = 0.4, tested on 18,640 pairs.

### Confusion Matrix

| | Predicted Same | Predicted Diff |
|---|----------------|----------------|
| **Actual Same** | 6,596 (TP) | 224 (FN) |
| **Actual Diff** | 0 (FP) | 11,820 (TN) |

### Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.8% |
| **Precision** | 100.0% |
| **Recall** | 96.7% |
| False Positives | 0 |
| False Negatives | 224 |

---

## 3. Classification Head Pipeline

Architecture: `Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→80)`

### Training

| Split | Samples |
|-------|---------|
| Train | 2,471 |
| Validation | 529 |
| Test | 530 |

### Results

| Metric | Value |
|--------|-------|
| **Train Accuracy** | 100.00% |
| **Validation Accuracy** | 99.81% |
| **Test Accuracy (Top-1)** | 99.06% |
| **Test Accuracy (Top-5)** | 99.06% |

### Inference on 100k Images

| Confidence Level | Count | Percent |
|-----------------|-------|---------|
| High (>0.9) | 2,876 | 2.88% |
| Medium (0.5-0.9) | 568 | 0.57% |
| Low (<0.5) | 96,556 | 96.56% |

*Note: Low confidence expected on megaface distractors (different identities)*

---

## Summary Table

| Pipeline | Task | Metric | Value |
|----------|------|--------|-------|
| Embedding Similarity | Verification | ROC-AUC | **0.9953** |
| Embedding Similarity | Verification | Best Accuracy | **99.58%** |
| Embedding Similarity | Identification | Rank-1 | **99.29%** |
| Pair Verification | Classification | Accuracy | **98.8%** |
| Pair Verification | Classification | Precision | **100.0%** |
| Classification Head | Identity | Top-1 Accuracy | **99.06%** |

---

## Files Generated

| File | Description |
|------|-------------|
| `megaface/feature_out_100k/features.list` | 100k embeddings (966 MB) |
| `megaface/baseline_metrics.txt` | Verification/Identification metrics |
| `megaface/pairs_data/test_pairs.json` | 18,640 test pairs |
| `megaface/error_visualization/` | Error analysis with images |
| `megaface/classification_output/classification_head.pth` | Trained model |
| `megaface/classification_output/all_100k_predictions.txt` | All predictions |

---

## Commands to Reproduce

```bash
# 1. Extract embeddings
cd MagFace/inference
../../venv/bin/python gen_feat.py --arch iresnet100 \
    --inf_list /path/to/images.list \
    --feat_list /path/to/features.list \
    --resume magface_epoch_00025.pth \
    --embedding_size 512

# 2. Compute baseline metrics
./venv/bin/python compute_baseline_metrics.py \
    --features megaface/feature_out_100k/features.list

# 3. Run pair verification
./venv/bin/python visualize_errors.py \
    --model magface_epoch_00025.pth \
    --feedback_file megaface/pairs_data/test_pairs.json \
    --threshold 0.4

# 4. Train classification head
./venv/bin/python classification_head_pipeline.py \
    --features megaface/feature_out_100k/features.list \
    --epochs 30
```
