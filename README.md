# MagFace Active Learning for Face Recognition

Active learning system for face recognition using MagFace with user feedback and fine-tuning.

**For detailed technical explanation, see [CONTENT.md](CONTENT.md)**

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
  - [1. Generate Test Dataset](#1-generate-test-dataset)
  - [2. Run Inference & Visualize Errors](#2-run-inference--visualize-errors)
  - [3. Fine-Tune on Feedback](#3-fine-tune-on-feedback)
  - [4. Evaluate Fine-Tuned Model](#4-evaluate-fine-tuned-model)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start (Megaface Workflow, GPU)

Use these commands to run MagFace inference, build hard pairs, visualize errors, and fine-tune:

```bash
# 1) Activate environment
conda activate magface

# 2) 150k random Megaface images -> embeddings
shuf -n 150000 megaface/data/megaface_abs.list > megaface/data/megaface_150k.list
mkdir -p megaface/feature_out_150k
PYTHONPATH=MagFace_repo CUDA_VISIBLE_DEVICES=0 python MagFace_repo/inference/gen_feat.py \
  --arch iresnet100 \
  --inf_list megaface/data/megaface_150k.list \
  --feat_list megaface/feature_out_150k/features.list \
  --resume magface_epoch_00025.pth \
  --batch_size 128 \
  --workers 8

# 3) Create hard pairs from FaceScrub embeddings (example: 12k total)
# Update pos_target / neg_target for different sizes
python - <<'PY'
import json, random, numpy as np
from collections import defaultdict
random.seed(42); np.random.seed(42)
features_file = 'megaface/feature_out/facescrub_features.list'
out_path = 'megaface/pairs_data/hard_pairs_12k.json'
pos_target, neg_target, candidate_mult = 6000, 6000, 6
paths, embs = [], []
with open(features_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 513:
            continue
        paths.append(parts[0])
        embs.append(np.array(list(map(float, parts[1:])), dtype=np.float32))
embs = np.vstack(embs)
embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
id_to_idxs = defaultdict(list)
for idx, p in enumerate(paths):
    identity = p.split('/')[-2]
    id_to_idxs[identity].append(idx)
ids = [i for i, idxs in id_to_idxs.items() if len(idxs) >= 2]
pos_candidates, neg_candidates = [], []
pos_needed, neg_needed = pos_target * candidate_mult, neg_target * candidate_mult
while len(pos_candidates) < pos_needed:
    identity = random.choice(ids)
    i, j = random.sample(id_to_idxs[identity], 2)
    sim = float(np.dot(embs[i], embs[j]))
    pos_candidates.append((sim, i, j))
all_indices = list(range(len(paths)))
while len(neg_candidates) < neg_needed:
    i, j = random.sample(all_indices, 2)
    if paths[i].split('/')[-2] == paths[j].split('/')[-2]:
        continue
    sim = float(np.dot(embs[i], embs[j]))
    neg_candidates.append((sim, i, j))
pos_candidates.sort(key=lambda x: x[0])
neg_candidates.sort(key=lambda x: x[0], reverse=True)
pairs = []
for _, i, j in pos_candidates[:pos_target]:
    pairs.append({'image1': paths[i], 'image2': paths[j], 'label': 1})
for _, i, j in neg_candidates[:neg_target]:
    pairs.append({'image1': paths[i], 'image2': paths[j], 'label': 0})
random.shuffle(pairs)
with open(out_path, 'w') as f:
    json.dump({'pairs': pairs}, f, indent=2)
print('Saved', out_path, 'total:', len(pairs))
PY

# 4) Visualize errors (fast: uses cached embeddings)
CUDA_VISIBLE_DEVICES=0 python visualize_errors.py \
  --model magface_epoch_00025.pth \
  --feedback_file megaface/pairs_data/hard_pairs_12k.json \
  --threshold 0.4 \
  --output_dir megaface/error_vis_hard_12k \
  --features_list megaface/feature_out/facescrub_features.list \
  --max_fp 500 \
  --max_fn 500

# 5) Fine-tune on hard pairs
CUDA_VISIBLE_DEVICES=0 python finetune_on_feedback.py \
  --checkpoint magface_epoch_00025.pth \
  --feedback_file megaface/pairs_data/hard_pairs_12k.json \
  --output_dir checkpoints_feedback_hard_12k \
  --epochs 5 \
  --batch_size 32 \
  --lr 0.00001 \
  --margin 0.4 \
  --val_split 0.2 \
  --seed 42
```

**Note:** FaceScrub contains ~3.5k labeled images, so large pair sets reuse images.
---

## 🔧 Environment Setup

### Prerequisites

- Python 3.9
- Anaconda/Miniconda

### Installation

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate magface
```

**Note:** Use `--force_cpu` if CUDA is not available.

---

## 📁 Project Structure

```
magface-active-learning/
├── magface_epoch_00025.pth          # Pretrained model (270MB)
├── megaface/                        # Megaface + FaceScrub data + outputs
│   ├── data/
│   │   ├── facescrub_images/
│   │   ├── megaface_images/
│   │   └── *.list
│   ├── feature_out*/                # Embeddings
│   ├── pairs_data/                  # Pair JSONs
│   └── error_vis*/                  # Visualizations
├── checkpoints_feedback*/           # Fine-tuned models
├── create_test_feedback.py          # CASIA toy dataset generator
├── visualize_errors.py              # Run inference & visualize
├── finetune_on_feedback.py          # Fine-tune model
├── evaluate_finetuned_model.py      # Evaluation script
├── environment.yml                  # Conda environment
├── README.md                        # This file
└── CONTENT.md                       # Technical documentation
```

---

## 📂 Required Files

### Dataset

- **Megaface + FaceScrub**: `megaface/data/`
- **CASIA-WebFace (optional toy workflow)**: `faces_webface_112x112/`
- **Pretrained Model**: `magface_epoch_00025.pth` (270MB)

### Generated Files

- **Test Dataset**: `test_feedback/` (created by script)
- **Fine-tuned Models**: `checkpoints_feedback/` (created during training)
- **Megaface Embeddings**: `megaface/feature_out*/` (created by inference)
- **Pair JSONs**: `megaface/pairs_data/` (hard/balanced pairs)
- **Error Visualizations**: `megaface/error_vis*/` and other output dirs

---

## 📖 How to Run

### 1. Generate Test Dataset (CASIA Toy Workflow)

**Create test image pairs for evaluation:**

```bash
python create_test_feedback.py --dataset faces_webface_112x112
```

**Output:**
- `test_feedback/feedback_pairs_test.json` - 200 image pairs
- `test_feedback/images/` - Extracted face images

**Parameters:**
- `--dataset`: Path to CASIA-WebFace dataset
- `--num_identities`: Number of people (default: 10)
- `--images_per_identity`: Images per person (default: 5)
- `--num_pairs`: Total pairs to generate (default: 200)

---

### 2. Run Inference & Visualize Errors (CASIA Toy Workflow)

**Run MagFace on test pairs and save error visualizations:**

```bash
python visualize_errors.py \
    --model magface_epoch_00025.pth \
    --feedback_file test_feedback/feedback_pairs_test.json \
    --threshold 0.4 \
    --force_cpu \
    --output_dir magface_errors
```

**Output:**
- `magface_errors/false_positives/` - Wrong "same" predictions
- `magface_errors/false_negatives/` - Wrong "different" predictions
- `magface_errors/error_summary.json` - Statistics

**Parameters:**
- `--model`: Path to model checkpoint
- `--feedback_file`: Path to test pairs JSON
- `--threshold`: Similarity threshold (default: 0.4)
- `--features_list`: Optional features.list to speed up evaluation
- `--max_fp`: Max false positives to save (default: all)
- `--max_fn`: Max false negatives to save (default: all)
- `--force_cpu`: Use CPU instead of GPU
- `--output_dir`: Where to save visualizations

---

### 3. Fine-Tune on Feedback (CASIA Toy Workflow)

**Train the model on user feedback pairs:**

```bash
python finetune_on_feedback.py \
    --checkpoint magface_epoch_00025.pth \
    --feedback_file test_feedback/feedback_pairs_test.json \
    --epochs 10 \
    --batch_size 16 \
    --force_cpu
```

**Output:**
- `checkpoints_feedback/magface_feedback_epoch1.pth`
- `checkpoints_feedback/magface_feedback_best.pth` - Best model

**Parameters:**
- `--checkpoint`: Pretrained model to start from
- `--feedback_file`: Training pairs JSON
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.0001)
- `--margin`: Contrastive loss margin (default: 0.4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--force_cpu`: Use CPU mode

**Training Time:** ~10-15 minutes per epoch on CPU

---

### 4. Evaluate Fine-Tuned Model (CASIA Toy Workflow)

**Test the fine-tuned model on same test pairs:**

```bash
python visualize_errors.py \
    --model checkpoints_feedback/magface_feedback_best.pth \
    --feedback_file test_feedback/feedback_pairs_test.json \
    --threshold 0.4 \
    --force_cpu \
    --output_dir finetuned_errors
```

**Compare Results:**
- Check `magface_errors/` (pretrained) vs `finetuned_errors/` (fine-tuned)
- Look at `error_summary.json` in each folder
- See if false positives/negatives decreased

---

## 🎯 What This Does

- **Inference**: Run MagFace face recognition on image pairs
- **Error Visualization**: See which pairs the model gets wrong
- **Fine-Tuning**: Improve the model using user feedback
- **Evaluation**: Measure accuracy and performance

---

## 🔧 Troubleshooting

### Common Issues

**GPU Not Working:**
- Use `--force_cpu` flag for all scripts
- Check `nvidia-smi` and CUDA driver compatibility

**Out of Memory:**
- Reduce `--batch_size` (try 8 or 4)
- Close other applications

**Slow Training:**
- Normal on CPU (~10-15 min/epoch)
- Reduce `--epochs` for testing

**File Not Found:**
- Check dataset path: `faces_webface_112x112/`
- Check Megaface data path: `megaface/data/`
- Check model path: `magface_epoch_00025.pth`
- Run `create_test_feedback.py` first

---

## 📊 Understanding Output

### Error Visualization Folders

**Each error case contains:**
- `img1_*.jpg` - First image
- `img2_*.jpg` - Second image  
- `comparison.png` - Side-by-side visualization
- `info.txt` - Similarity score and labels

**Error Types:**
- **False Positive**: Model said "same" but actually different people
- **False Negative**: Model said "different" but actually same person
- **True Positive**: Correctly identified same person
- **True Negative**: Correctly identified different people

### Error Summary JSON

```json
{
  "accuracy": 0.88,
  "true_positives": 76,
  "true_negatives": 100,
  "false_positives": 0,
  "false_negatives": 24,
  "precision": 1.0,
  "recall": 0.76
}
```

---

## 📚 References

- **MagFace Paper:** [MagFace: A Universal Representation for Face Recognition and Quality Assessment](https://arxiv.org/abs/2103.06627)
- **CASIA-WebFace Dataset:** 10,575 identities, 494,414 images
- **Technical Details:** See [CONTENT.md](CONTENT.md) for complete explanation

---

**For detailed technical documentation, bug fixes, and architecture details, see [CONTENT.md](CONTENT.md)**
