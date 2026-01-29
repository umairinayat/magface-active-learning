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

## 🚀 Quick Start

**✅ Your Current Scripts Are Better:**

Use these **corrected scripts** with proper preprocessing and cosine similarity:

```bash
# 1. Activate environment
conda activate magface

# 2. Generate test dataset
python create_test_feedback.py --dataset faces_webface_112x112

# 3. Run inference and visualize errors
python visualize_errors.py \
    --model magface_epoch_00025.pth \
    --feedback_file test_feedback/feedback_pairs_test.json \
    --threshold 0.4 \
    --force_cpu \
    --output_dir magface_errors

# 4. Fine-tune on feedback
python finetune_on_feedback.py \
    --checkpoint magface_epoch_00025.pth \
    --feedback_file test_feedback/feedback_pairs_test.json \
    --epochs 10 \
    --batch_size 16 \
    --force_cpu


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

**Note:** Use `--force_cpu` flag for all scripts (GPU compatibility issues)

---

## 📁 Project Structure

```
mag_face_testing/
├── magface_epoch_00025.pth          # Pretrained model (270MB)
├── faces_webface_112x112/           # CASIA-WebFace dataset (2.7GB)
├── test_feedback/                   # Generated test data
│   ├── feedback_pairs_test.json
│   ├── ground_truth.json
│   └── images/
├── checkpoints_feedback/            # Fine-tuned models
│   ├── magface_feedback_epoch*.pth
│   └── magface_feedback_best.pth
├── create_test_feedback.py          # Generate test dataset
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

- **CASIA-WebFace**: `faces_webface_112x112/` (2.7GB)
- **Pretrained Model**: `magface_epoch_00025.pth` (270MB)

### Generated Files

- **Test Dataset**: `test_feedback/` (created by script)
- **Fine-tuned Models**: `checkpoints_feedback/` (created during training)
- **Error Visualizations**: Output directories (created by script)

---

## 📖 How to Run

### 1. Generate Test Dataset

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

### 2. Run Inference & Visualize Errors

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
- `--force_cpu`: Use CPU instead of GPU
- `--output_dir`: Where to save visualizations

---

### 3. Fine-Tune on Feedback

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
- `--margin`: Contrastive loss margin (default: 1.0)
- `--force_cpu`: Use CPU mode

**Training Time:** ~10-15 minutes per epoch on CPU

---

### 4. Evaluate Fine-Tuned Model

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
- RTX 5050 not supported by current PyTorch version

**Out of Memory:**
- Reduce `--batch_size` (try 8 or 4)
- Close other applications

**Slow Training:**
- Normal on CPU (~10-15 min/epoch)
- Reduce `--epochs` for testing

**File Not Found:**
- Check dataset path: `faces_webface_112x112/`
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
