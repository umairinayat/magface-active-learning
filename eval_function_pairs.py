#!/usr/bin/env python
"""
Evaluate a MagFace embedding model on labeled image pairs.

- If model_path is not provided, uses config.yaml -> model.magface.weights
- Uses the same preprocessing + cosine similarity as inference_with_embedding.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from img_embedding import (
    _load_yaml_config,
    _resolve_weights_path,
    _build_model,
    _load_weights,
    _normalize_label,
)


def _get_threshold(cfg: Dict[str, Any], override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    matching = cfg.get("matching", {})
    return float(matching.get("similarity_threshold", 0.4))


def _get_model_cfg(cfg: Dict[str, Any]) -> Tuple[str, str, int, str]:
    model_cfg = cfg.get("model", {})
    magface_cfg = model_cfg.get("magface", {})
    weights = magface_cfg.get("weights", "magface_epoch_00025.pth")
    arch = magface_cfg.get("arch", "iresnet100")
    embedding_size = int(magface_cfg.get("embedding_size", 512))
    device = model_cfg.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return weights, arch, embedding_size, device


def _preprocess(path: str, transform: T.Compose) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img).unsqueeze(0)


def _load_model(
    model_path: Optional[str],
    config_path: Optional[str],
    device: Optional[str],
) -> Tuple[torch.nn.Module, float]:
    cfg = _load_yaml_config(config_path)
    weights, arch, embedding_size, cfg_device = _get_model_cfg(cfg)
    threshold = _get_threshold(cfg, override=None)

    if model_path is None:
        model_path = _resolve_weights_path(weights, config_path)
    if device is None:
        device = cfg_device

    model = _build_model(arch, embedding_size)
    _load_weights(model, model_path)
    model.to(device)
    model.eval()
    return model, threshold


def _extract_embedding(
    model: torch.nn.Module,
    image_path: str,
    device: str,
    transform: T.Compose,
    cache: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if cache is not None and image_path in cache:
        return cache[image_path]

    tensor = _preprocess(image_path, transform).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = F.normalize(emb, p=2, dim=1)
    vec = emb.cpu().numpy()[0]

    if cache is not None:
        cache[image_path] = vec
    return vec


def evaluate_pairs(
    pairs: List[Union[Dict[str, Any], List[Any], Tuple[Any, Any, Any]]],
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    threshold: Optional[float] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on labeled image pairs.

    pairs can be:
      - dicts: {"image1": ..., "image2": ..., "label": ...}
      - lists/tuples: [image1, image2, label]
    """
    cfg = _load_yaml_config(config_path)
    model, cfg_threshold = _load_model(model_path, config_path, device)
    if device is None:
        device = next(model.parameters()).device.type
    thr = _get_threshold(cfg, override=threshold) if threshold is not None else cfg_threshold

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
    ])

    cache = {} if use_cache else None

    tp = tn = fp = fn = 0
    sims = []

    for p in pairs:
        if isinstance(p, dict):
            img1 = p["image1"]
            img2 = p["image2"]
            label = _normalize_label(p["label"])
        else:
            img1, img2, label_raw = p
            label = _normalize_label(label_raw)

        emb1 = _extract_embedding(model, img1, device, transform, cache=cache)
        emb2 = _extract_embedding(model, img2, device, transform, cache=cache)
        sim = float(np.dot(emb1, emb2))
        sims.append(sim)

        pred = 1.0 if sim > thr else 0.0
        if pred == 1.0 and label == 1.0:
            tp += 1
        elif pred == 0.0 and label == 0.0:
            tn += 1
        elif pred == 1.0 and label == 0.0:
            fp += 1
        elif pred == 0.0 and label == 1.0:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "threshold": thr,
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "similarity_mean": float(np.mean(sims)) if sims else 0.0,
        "similarity_std": float(np.std(sims)) if sims else 0.0,
    }


def eval_function(
    pairs: List[Union[Dict[str, Any], List[Any], Tuple[Any, Any, Any]]],
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    threshold: Optional[float] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Alias for evaluate_pairs (requested name)."""
    return evaluate_pairs(
        pairs=pairs,
        model_path=model_path,
        config_path=config_path,
        threshold=threshold,
        device=device,
        use_cache=use_cache,
    )

