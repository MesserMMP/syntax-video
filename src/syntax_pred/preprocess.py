# src/syntax_pred/preprocess.py — чтение DICOM и тестовый пайплайн трансформаций

from __future__ import annotations
from typing import Tuple
import os
import numpy as np
import torch
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize
import pydicom

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

def read_dicom_uint8(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array  # (T,H,W) или (H,W)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    if arr.dtype == np.uint16:
        vmax = float(arr.max()) or 1.0
        arr = (arr.astype(np.float32) * (255.0 / vmax)).astype(np.uint8)
    if arr.dtype != np.uint8:
        raise TypeError(f"Expected uint8 after conversion, got {arr.dtype} for {os.path.basename(path)}")
    return arr

def ensure_length_center_crop(arr: np.ndarray, frames: int) -> np.ndarray:
    t_now = arr.shape[0]
    while t_now < frames:
        arr = np.concatenate([arr, arr], axis=0)
        t_now = arr.shape[0]
    start = (t_now - frames) // 2
    return arr[start:start + frames]

def test_like_transform(video_size: Tuple[int, int]) -> T.Compose:
    return T.Compose([
        ToTensorVideo(),
        T.Resize(size=video_size, antialias=True),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
