from __future__ import annotations
from typing import List, Tuple, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import (
    read_dicom_uint8,
    ensure_length_center_crop,
    test_like_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

DTYPE = torch.float16

class SyntaxInferenceDataset(Dataset):
    """
    Загрузчик DICOM для инференса одной артерии.
    Возвращает:
      videos: (S=1, C, T, H, W) — один клип на элемент, RGB получен дублированием серого канала
      label:  заглушка (torch.tensor([0], dtype=DTYPE))
      target: заглушка (torch.tensor([0.0], dtype=DTYPE))
      uid:    имя файла
    """
    def __init__(
        self,
        files: List[str],
        frames_per_clip: int,
        video_size: Tuple[int, int],
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
        transform: Optional[torch.nn.Module] = None,
    ):
        self.files = list(files)
        self.frames = int(frames_per_clip)
        self.video_size = video_size
        self.mean = mean
        self.std = std
        self.transform = transform or test_like_transform(video_size)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        uid = os.path.basename(path)

        # (T,H,W) uint8 → выравнивание длины
        arr = read_dicom_uint8(path)
        arr = ensure_length_center_crop(arr, self.frames)

        # (T,H,W,3): серый → RGB
        vid_thwc = np.stack([arr, arr, arr], axis=-1)
        vid_thwc = torch.tensor(vid_thwc)

        # (C,T,H,W) → добавляем размерность последовательности S=1
        vid_cthw = self.transform(vid_thwc)
        videos = vid_cthw.unsqueeze(0)

        label = torch.tensor([0], dtype=DTYPE)
        target = torch.tensor([0.0], dtype=DTYPE)
        return videos, label, target, uid
