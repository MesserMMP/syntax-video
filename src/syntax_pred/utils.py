from __future__ import annotations
import glob, os
import torch

def pick_device(pref: str = "auto") -> torch.device:
    """
    Выбор устройства для вычислений: CUDA → MPS → CPU.
    При явном указании pref возвращается соответствующее устройство.
    """
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)

def discover_weights(dirpath: str) -> list[str]:
    """
    Поиск весов моделей по маскам *.pt и *.ckpt внутри указанной директории.
    Возвращает отсортированный список путей.
    """
    pats = [os.path.join(dirpath, "*.pt"), os.path.join(dirpath, "*.ckpt")]
    found = []
    for p in pats:
        found.extend(glob.glob(p))
    return sorted(found)
