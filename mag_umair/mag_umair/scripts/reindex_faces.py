#!/usr/bin/env python
"""
Re-index All Faces with New Model

After fine-tuning, run this script to:
1. Clear old embeddings from Qdrant
2. Generate new embeddings with the fine-tuned model
3. Store new embeddings in Qdrant

Usage:
    python scripts/reindex_faces.py --images_dir path/to/faces/
    
Or to reindex from a list of image paths:
    python scripts/reindex_faces.py --images_list paths.txt
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from facelib import FaceLibrary


def main():
    parser = argparse.ArgumentParser(description="Re-index faces with new model")
    
    parser.add_argument("--images_dir", default=None,
                        help="Directory containing face images to re-index")
    parser.add_argument("--images_list", default=None,
                        help="Text file with image paths (one per line)")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (optional)")
    parser.add_argument("--clear", action="store_true", default=True,
                        help="Clear existing index before re-indexing (default: True)")
    parser.add_argument("--no-clear", dest="clear", action="store_false",
                        help="Don't clear existing index")
    
    args = parser.parse_args()
    
    if not args.images_dir and not args.images_list:
        parser.error("You must specify either --images_dir or --images_list")
    
    # Initialize FaceLibrary (loads model from config.yaml)
    print("=" * 60)
    print("RE-INDEXING FACES WITH NEW MODEL")
    print("=" * 60)
    
    lib = FaceLibrary(config_path=args.config)
    
    # Show current model
    print(f"\nModel loaded successfully!")
    print(f"Current index size: {lib.size} faces")
    
    # Clear old embeddings
    if args.clear:
        print(f"\n[Step 1] Clearing old embeddings...")
        lib.face_index.clear()
        print(f" Old embeddings cleared. Index size: {lib.size}")
    
    # Collect image paths
    image_paths = []
    
    if args.images_dir:
        images_path = Path(args.images_dir)
        if not images_path.exists():
            print(f" Directory not found: {args.images_dir}")
            return
        
        # Find all images
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        for ext in extensions:
            image_paths.extend(images_path.rglob(f"*{ext}"))
            image_paths.extend(images_path.rglob(f"*{ext.upper()}"))
        
        print(f"\n[Step 2] Found {len(image_paths)} images in {args.images_dir}")
    
    elif args.images_list:
        with open(args.images_list, 'r') as f:
            image_paths = [Path(line.strip()) for line in f if line.strip()]
        print(f"\n[Step 2] Found {len(image_paths)} images in list file")
    
    if not image_paths:
        print(" No images found!")
        return
    
    # Re-index all images
    print(f"\n[Step 3] Generating new embeddings with fine-tuned model...")
    
    success_count = 0
    fail_count = 0
    
    for img_path in tqdm(image_paths, desc="Indexing"):
        try:
            face_ids = lib.index(str(img_path))
            if face_ids:
                success_count += len(face_ids)
        except Exception as e:
            fail_count += 1
            # Uncomment to see errors:
            # print(f"  Failed: {img_path} - {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("RE-INDEXING COMPLETE")
    print("=" * 60)
    print(f" Successfully indexed: {success_count} faces")
    print(f" Failed: {fail_count} images")
    print(f" Total index size: {lib.size} faces")
    print("=" * 60)
    
    print("\n Your database now uses embeddings from the fine-tuned model!")
    print("   All searches will use the improved embeddings.")


if __name__ == "__main__":
    main()
