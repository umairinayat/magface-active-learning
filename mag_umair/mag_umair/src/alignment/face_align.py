"""
Face alignment using 5-point landmarks.
Based on ArcFace alignment (adapted from insightface/utils/face_align.py)
License: MIT (alignment logic is standard computer vision)
"""
import cv2
import numpy as np
from skimage import transform as trans
from typing import Tuple, Optional


# ArcFace reference landmarks for 112x112 output
# Order: left_eye, right_eye, nose, mouth_left, mouth_right
ARCFACE_DST = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # mouth left
    [70.7299, 92.2041]   # mouth right
], dtype=np.float32)


def estimate_transform(landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """
    Estimate similarity transform matrix from 5-point landmarks.
    
    Args:
        landmarks: 5-point landmarks shape (5, 2)
        output_size: Target output size (default 112 for ArcFace/MagFace)
    
    Returns:
        Affine transform matrix (2, 3)
    """
    assert landmarks.shape == (5, 2), f"Expected (5, 2), got {landmarks.shape}"
    
    # Scale reference landmarks to output size
    if output_size % 112 == 0:
        ratio = float(output_size) / 112.0
        diff_x = 0
    else:
        ratio = float(output_size) / 128.0
        diff_x = 8.0 * ratio
    
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    
    # Estimate similarity transform
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    M = tform.params[0:2, :]
    
    return M


def align_face(
    img: np.ndarray, 
    landmarks: np.ndarray, 
    output_size: int = 112,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    margin: float = 0.3,
    border_value: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Align face using similarity transform.
    
    Args:
        img: BGR image (OpenCV format)
        landmarks: 5-point landmarks shape (5, 2)
                   Order: left_eye, right_eye, nose, mouth_left, mouth_right
        output_size: Output image size (default 112 for MagFace)
        bbox: Optional face bounding box [x1, y1, x2, y2] for pre-cropping
        margin: Margin ratio to expand bbox before cropping (default 0.3)
        border_value: Border fill color
    
    Returns:
        Aligned face image (output_size x output_size x 3)
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        x1_expanded = max(0, int(x1 - w * margin))
        y1_expanded = max(0, int(y1 - h * margin))
        x2_expanded = min(img.shape[1], int(x2 + w * margin))
        y2_expanded = min(img.shape[0], int(y2 + h * margin))
        
        cropped = img[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        
        adjusted_landmarks = landmarks.copy()
        adjusted_landmarks[:, 0] -= x1_expanded
        adjusted_landmarks[:, 1] -= y1_expanded
        
        M = estimate_transform(adjusted_landmarks, output_size)
        aligned = cv2.warpAffine(
            cropped, M, (output_size, output_size), 
            borderValue=border_value
        )
    else:
        M = estimate_transform(landmarks, output_size)
        aligned = cv2.warpAffine(
            img, M, (output_size, output_size), 
            borderValue=border_value
        )
    
    return aligned


def align_face_with_transform(
    img: np.ndarray, 
    landmarks: np.ndarray, 
    output_size: int = 112
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align face and return both aligned image and transform matrix.
    
    Returns:
        (aligned_face, transform_matrix)
    """
    M = estimate_transform(landmarks, output_size)
    aligned = cv2.warpAffine(img, M, (output_size, output_size), borderValue=0.0)
    return aligned, M
