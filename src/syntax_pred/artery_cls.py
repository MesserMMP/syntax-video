# src/syntax_pred/artery_cls.py
from __future__ import annotations
from typing import List, Dict, Any
import os
import numpy as np
import torch
import torch.nn as nn
import pydicom
import torchvision.models.video as tvmv
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize

from .config import CFG, DEVICE
from .hf_weights import fetch_classifier_weight  # ← добавили импорт

_ARTERY_MODEL: nn.Module | None = None

def _resolve_classifier_weights_path() -> str:
    """
    1) Берём CFG.classifier.weights, если файл существует.
    2) Иначе — скачиваем из HF-репозитория (ENV WEIGHTS_REPO или CFG.weights_repo)
       из подпапки CFG.classifier.hf_subdir (по умолчанию 'classifier').
    """
    local_path = str(getattr(CFG.classifier, "weights", "") or "").strip()
    if local_path and os.path.isfile(local_path):
        return local_path

    repo_id = os.environ.get("WEIGHTS_REPO", getattr(CFG, "weights_repo", "") or "")
    subdir  = str(getattr(CFG.classifier, "hf_subdir", "classifier"))
    if repo_id:
        pulled = fetch_classifier_weight(repo_id=repo_id, subdir=subdir)
        if pulled and os.path.isfile(pulled):
            return pulled

    # не нашли — вернём что есть (вызовущий код выбросит понятную ошибку)
    return local_path

def _load_artery_model() -> nn.Module:
    global _ARTERY_MODEL
    if _ARTERY_MODEL is not None:
        return _ARTERY_MODEL

    weights_path = _resolve_classifier_weights_path()
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Classifier weights not found. "
            f"Set a valid path in configs/default.yaml: classifier.weights "
            f"or ensure HF repo '{getattr(CFG, 'weights_repo', '')}' has '{getattr(CFG.classifier, 'hf_subdir', 'classifier')}/*.pt|*.ckpt'."
        )

    model = tvmv.r3d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=1, bias=True)

    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if (isinstance(raw, dict) and "state_dict" in raw) else raw
    model_sd = {}
    for k, v in sd.items():
        assert k.startswith("model.")
        model_sd[k.split(".", 1)[1]] = v

    load_result = model.load_state_dict(model_sd, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print("[classifier] load warnings:", load_result)

    model.to(DEVICE).eval()
    _ARTERY_MODEL = model
    return model

def _artery_transform() -> T.Compose:
    video_size = tuple(CFG.classifier.video_size)
    mean = CFG.classifier.mean
    std = CFG.classifier.std
    return T.Compose([
        ToTensorVideo(),
        T.Resize(size=video_size),
        Normalize(mean=mean, std=std),
    ])

@torch.no_grad()
def classify_artery(files: List[str]) -> Dict[str, Any]:
    model = _load_artery_model()
    tx = _artery_transform()
    left_max = float(CFG.classifier.thresholds.left_max)
    right_min = float(CFG.classifier.thresholds.right_min)

    left, right, other = [], [], []
    for p in files:
        ds = pydicom.dcmread(p)
        arr = ds.pixel_array
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        if arr.dtype == np.uint16:
            vmax = float(arr.max()) or 1.0
            arr = (arr.astype(np.float32) * (255.0 / vmax)).astype(np.uint8)

        thwc = np.stack([arr, arr, arr], axis=-1)
        x = tx(torch.tensor(thwc)).unsqueeze(0).to(DEVICE)
        prob = float(torch.sigmoid(model(x))[0, 0].detach().cpu())
        rec = {"path": p, "artery_prob": prob}
        if prob <= left_max:
            left.append(rec)
        elif prob >= right_min:
            right.append(rec)
        else:
            other.append(rec)
    return {"left": left, "right": right, "other": other}
