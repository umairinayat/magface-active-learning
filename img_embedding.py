#!/usr/bin/env python
"""
Single-image embedding helper.

Usage:
    from img_embedding import get_embedding
    emb = get_embedding("path/to/image.jpg")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import cv2
import random

from MagFace_repo.models.iresnet import iresnet100, iresnet50, iresnet18


_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def _load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to read the config file. "
            "Install with: pip install pyyaml"
        ) from exc

    if config_path is None:
        candidates = [
            Path("config.yaml"),
        ]
        for p in candidates:
            if p.exists():
                config_path = str(p)
                break

    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _resolve_weights_path(weights: str, config_path: Optional[str]) -> str:
    if os.path.isabs(weights):
        return weights
    if config_path:
        base = Path(config_path).parent
        candidate = base / weights
        if candidate.exists():
            return str(candidate)
    if Path(weights).exists():
        return weights
    # Fallback: try basename in cwd
    fallback = Path(weights).name
    if Path(fallback).exists():
        return str(Path(fallback))
    raise FileNotFoundError(f"Model weights not found: {weights}")


def _build_model(arch: str, embedding_size: int) -> torch.nn.Module:
    if arch == "iresnet100":
        return iresnet100(pretrained=False, num_classes=embedding_size)
    if arch == "iresnet50":
        return iresnet50(pretrained=False, num_classes=embedding_size)
    if arch == "iresnet18":
        return iresnet18(pretrained=False, num_classes=embedding_size)
    raise ValueError(f"Unknown arch: {arch}. Use iresnet100/50/18.")


def _load_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned = {}
    model_dict = model.state_dict()

    for k, v in state_dict.items():
        if k.startswith("features.module."):
            new_k = k.replace("features.module.", "")
        elif k.startswith("module.features."):
            new_k = k.replace("module.features.", "")
        elif k.startswith("module."):
            new_k = k.replace("module.", "")
        elif k.startswith("features."):
            new_k = k.replace("features.", "")
        else:
            new_k = k

        # Skip classification head if mismatched
        if new_k == "fc.weight" and v.shape[0] != model_dict[new_k].shape[0]:
            continue
        if new_k == "fc.bias" and v.shape[0] != model_dict[new_k].shape[0]:
            continue

        if new_k in model_dict and v.shape == model_dict[new_k].shape:
            cleaned[new_k] = v

    model.load_state_dict(cleaned, strict=False)


def _get_model(
    arch: str,
    embedding_size: int,
    weights_path: str,
    device: str,
) -> torch.nn.Module:
    cache_key = f"{arch}:{embedding_size}:{weights_path}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = _build_model(arch, embedding_size)
    _load_weights(model, weights_path)
    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _preprocess_image(image: Union[str, np.ndarray, Image.Image], assume_bgr: bool) -> torch.Tensor:
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if assume_bgr:
            arr = arr[..., ::-1]
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    else:
        raise TypeError("image must be a file path, PIL Image, or numpy array")

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),  # outputs [0, 1]
    ])
    return transform(img).unsqueeze(0)


def get_embedding(
    image: Union[str, np.ndarray, Image.Image],
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    assume_bgr: bool = False,
) -> np.ndarray:
    """
    Return L2-normalized embedding for a single image.

    - Reads model settings from config file (weights/arch/embedding_size).
    - Supports image path, PIL Image, or numpy array.
    - Returns a 512-dim float32 numpy vector (normalized).
    """
    cfg = _load_yaml_config(config_path)

    model_cfg = cfg.get("model", {})
    magface_cfg = model_cfg.get("magface", {})
    weights = magface_cfg.get("weights", "magface_epoch_00025.pth")
    arch = magface_cfg.get("arch", "iresnet100")
    embedding_size = magface_cfg.get("embedding_size", 512)

    if device is None:
        device = model_cfg.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path = _resolve_weights_path(weights, config_path)
    model = _get_model(arch, embedding_size, weights_path, device)

    tensor = _preprocess_image(image, assume_bgr=assume_bgr).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()[0].astype(np.float32)


def _normalize_label(label: Any) -> float:
    if isinstance(label, bool):
        return 1.0 if label else 0.0
    if isinstance(label, (int, np.integer)):
        return 1.0 if int(label) == 1 else 0.0
    if isinstance(label, (float, np.floating)):
        return 1.0 if float(label) >= 0.5 else 0.0
    if isinstance(label, str):
        s = label.strip().lower()
        if s in {"same", "positive", "pos", "true", "1", "yes"}:
            return 1.0
        if s in {"different", "diff", "negative", "neg", "false", "0", "no"}:
            return 0.0
    raise ValueError(f"Unsupported label value: {label}")


class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, Any]], transform: Optional[T.Compose] = None) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        img1_path = pair["image1"]
        img2_path = pair["image2"]

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Failed to read image(s): {img1_path}, {img2_path}")

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(pair["label"], dtype=torch.float32)
        return img1, img2, label


class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin: float = 0.4) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        similarity = torch.sum(emb1 * emb2, dim=1)
        loss_positive = label * torch.pow(1 - similarity, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(similarity - self.margin, min=0.0), 2)
        return torch.mean(loss_positive + loss_negative)


def _metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: str,
    threshold: float,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    tp = tn = fp = fn = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for img1, img2, label in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            emb1 = model(img1)
            emb2 = model(img2)

            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

            loss = criterion(emb1, emb2, label)
            total_loss += loss.item()

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            similarity = torch.sum(emb1 * emb2, dim=1)
            pred = (similarity > threshold).float()

            tp += int(((pred == 1) & (label == 1)).sum().item())
            tn += int(((pred == 0) & (label == 0)).sum().item())
            fp += int(((pred == 1) & (label == 0)).sum().item())
            fn += int(((pred == 0) & (label == 1)).sum().item())

    avg_loss = total_loss / max(1, len(loader))
    metrics = _metrics_from_counts(tp, tn, fp, fn)
    metrics.update({"loss": avg_loss, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return metrics


def finetune_pairs(
    pairs: List[Dict[str, Any]],
    config_path: Optional[str] = None,
    output_dir: str = "checkpoints_feedback_pairs",
    val_split: float = 0.2,
    seed: int = 42,
    device: Optional[str] = None,
    batch_size: int = 32,
    lr: float = 1e-5,
    epochs: int = 5,
    margin: float = 0.4,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Fine-tune MagFace on a list of labeled image pairs.

    Each pair dict must contain:
        - image1: str
        - image2: str
        - label: 1/0 or same/different strings
    """
    cfg = _load_yaml_config(config_path)
    model_cfg = cfg.get("model", {})
    magface_cfg = model_cfg.get("magface", {})

    weights = magface_cfg.get("weights", "magface_epoch_00025.pth")
    arch = magface_cfg.get("arch", "iresnet100")
    embedding_size = magface_cfg.get("embedding_size", 512)

    if device is None:
        device = model_cfg.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use explicit defaults to match training command values unless caller overrides.

    # Normalize labels and validate pairs
    norm_pairs = []
    for p in pairs:
        if "image1" not in p or "image2" not in p or "label" not in p:
            raise ValueError("Each pair must have image1, image2, and label keys.")
        norm_pairs.append({
            "image1": p["image1"],
            "image2": p["image2"],
            "label": _normalize_label(p["label"]),
        })

    # Split train/val
    rng = random.Random(seed)
    indices = list(range(len(norm_pairs)))
    rng.shuffle(indices)
    val_size = int(len(indices) * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_pairs = [norm_pairs[i] for i in train_idx]
    val_pairs = [norm_pairs[i] for i in val_idx]

    os.makedirs(output_dir, exist_ok=True)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
    ])

    train_ds = PairDataset(train_pairs, transform=transform)
    val_ds = PairDataset(val_pairs, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    weights_path = _resolve_weights_path(weights, config_path)
    model = _build_model(arch, embedding_size)
    _load_weights(model, weights_path)
    model.to(device)

    criterion = CosineSimilarityLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Baseline eval
    baseline = _run_epoch(model, val_loader, criterion, optimizer=None, device=device, threshold=margin, train=False)

    best_val_acc = baseline["accuracy"]
    best_path = os.path.join(output_dir, "magface_feedback_best.pth")

    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device, threshold=margin, train=True
        )
        val_metrics = _run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, threshold=margin, train=False
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)

    return {
        "baseline": baseline,
        "history": history,
        "best": {"val_accuracy": best_val_acc, "path": best_path},
        "sizes": {"train": len(train_pairs), "val": len(val_pairs)},
    }
