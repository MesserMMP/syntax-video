# src/syntax_pred/hf_weights.py
from __future__ import annotations
import os, glob
from typing import Dict, List, Iterable
from huggingface_hub import snapshot_download

def _collect(dirpath: str, patterns: Iterable[str] = ("*.pt", "*.ckpt")) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(dirpath, pat)))
    return sorted(out)

def fetch_weights(repo_id: str,
                  allow_patterns: Iterable[str] = ("left/*.pt","left/*.ckpt","right/*.pt","right/*.ckpt")
                 ) -> Dict[str, List[str]]:
    cache_dir = snapshot_download(repo_id=repo_id, allow_patterns=list(allow_patterns))
    left_dir  = os.path.join(cache_dir, "left")
    right_dir = os.path.join(cache_dir, "right")
    return {
        "left": _collect(left_dir)  if os.path.isdir(left_dir)  else [],
        "right": _collect(right_dir) if os.path.isdir(right_dir) else [],
    }

def fetch_classifier_weight(repo_id: str,
                            subdir: str = "classifier",
                            file_patterns: Iterable[str] = ("*.pt", "*.ckpt")) -> str:
    """
    Скачивает веса классификатора из HF-репозитория.
    Возвращает путь к первому (отсортированному) найденному файлу или "".
    """
    allow = [f"{subdir}/{p}" for p in file_patterns]
    cache_dir = snapshot_download(repo_id=repo_id, allow_patterns=allow)
    target_dir = os.path.join(cache_dir, subdir)
    files = _collect(target_dir, patterns=file_patterns) if os.path.isdir(target_dir) else []
    return files[0] if files else ""
