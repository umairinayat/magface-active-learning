#!/usr/bin/env python3
"""Test script to debug indexing issue with one_image directory."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from facelib import FaceLibrary, NoFaceDetectedError, FilterRejectedError

def test_indexing():
    print("=" * 60)
    print("INDEXING DEBUG TEST")
    print("=" * 60)
    
    # Test directory
    test_dir = project_root / "data" / "one_image"
    print(f"\n1. Test directory: {test_dir}")
    print(f"   Exists: {test_dir.exists()}")
    
    # List files
    if test_dir.exists():
        files = list(test_dir.iterdir())
        print(f"   Files found: {len(files)}")
        for f in files:
            print(f"     - {f.name} ({f.suffix.lower()})")
    
    # Check extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = [f for f in test_dir.rglob('*') if f.suffix.lower() in image_extensions]
    print(f"\n2. Image files matching extensions: {len(image_files)}")
    for f in image_files:
        print(f"     - {f}")
    
    # Initialize library
    print("\n3. Initializing FaceLibrary...")
    lib = FaceLibrary(debug_save_faces=True)
    print(f"   Initial index size: {lib.size}")
    
    # Try to index each file directly
    print("\n4. Testing direct indexing of each file...")
    for img_path in image_files:
        print(f"\n   Processing: {img_path.name}")
        try:
            # First test detection
            print("   - Detecting faces...")
            faces = lib.detect_faces(str(img_path))
            print(f"   - Detected {len(faces)} faces")
            for i, face in enumerate(faces):
                print(f"     Face {i}: bbox={face.bbox}, det_score={face.det_score:.3f}, quality={face.embedding_norm:.2f}")
            
            # Now try indexing
            print("   - Indexing...")
            face_ids = lib.index(str(img_path), image_id=img_path.stem)
            print(f"   - Indexed {len(face_ids)} faces: {face_ids}")
            
        except NoFaceDetectedError as e:
            print(f"   - ERROR: No face detected - {e}")
        except FilterRejectedError as e:
            print(f"   - ERROR: Filter rejected - {e}")
        except Exception as e:
            print(f"   - ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n5. Final index size: {lib.size}")
    
    # Test search if we have indexed faces
    if lib.size > 0:
        print("\n6. Testing search with same image...")
        try:
            results = lib.search(str(image_files[0]), top_k=5)
            print(f"   Found {len(results)} results:")
            for r in results:
                print(f"     - face_id={r.face_id}, score={r.score:.3f}, bbox={r.bbox}")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_indexing()
