# FaceLib - Face Recognition Library

A production-ready face recognition library built with MagFace and RetinaFace.

## Features

- **Face Detection**
- **Face Embedding**
- **Face Alignment**
- **Indexing & Search**
- **Quality Filtering**
- **Anime/Cartoon Detection**

## Installation

```bash
pip install git+https://github.com/iammuhammadamir/Facial_Matching_System_Dan_V2.git@v1.2.0
```

### Download MagFace Weights

Download the pretrained MagFace model weights (required):

1. Download from: https://drive.google.com/drive/folders/1Bd86upLze2H-a-H3q0bG1EuSB2wQU5YD
2. Place `magface_epoch_00025.pth` in `models/` directory

## Quick Start

```python
from facelib import FaceLibrary

# Initialize library
lib = FaceLibrary()

# Detect faces in an image
faces = lib.detect_faces("image.jpg")
print(f"Detected {len(faces)} faces")

# Index faces from images
face_ids = lib.index("person1.jpg")
face_ids = lib.index("person2.jpg")

# Search for similar faces
results = lib.search("query.jpg", top_k=5)
for r in results:
    print(f"Match: {r.image_id}, Score: {r.score:.3f}")

# Save index
lib.save()
```

## Configuration

Edit `config.yaml` to customize:

- Detection thresholds
- Quality filters (min face size, detection score, embedding quality)
- Qdrant connection (server URL, API key, collection name)
- Anime classifier settings

## Web Demo

Run the included web demo:

```bash

python demo/backend/app_facelib.py
```

Then open: http://localhost:8000/demo

## Architecture

```
Input Image → RetinaFace Detection → 5-Point Alignment → MagFace Embedding → Qdrant Search
                    ↓                       ↓                    ↓
              bbox, landmarks          112×112 face        512-dim vector
              det_score                                    + magnitude (quality)
```

## License

- **FaceLib**: MIT License
- **MagFace**: Apache 2.0 License
- **RetinaFace**: MIT License
- **Qdrant**: Apache 2.0 License

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- TensorFlow >= 2.10.0
- Qdrant (local Docker or cloud instance)
- OpenCV

See `pyproject.toml` for full dependency list.

## Development

```bash
# Clone repository
git clone https://github.com/iammuhammadamir/Facial_Matching_System_Dan_V2.git
cd Facial_Matching_System_Dan_V2

# Create virtual environment
python3.11 -m venv facelib-env
source facelib-env/bin/activate

# Install in development mode
pip install -e .

# Download MagFace weights to models/
```

## Version

Current version: **1.2.0**
