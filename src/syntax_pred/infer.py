# src/syntax_pred/infer.py — инференс SYNTAX (среднее по моделям), сортировка, Study

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import os
import numpy as np
import torch
import pydicom

from .config import CFG, DEVICE
from .utils import discover_weights
from .model import SyntaxLightningModule
from .preprocess import read_dicom_uint8, ensure_length_center_crop, test_like_transform
from .hf_weights import fetch_weights
from .artery_cls import classify_artery


# -------- Data structure --------
@dataclass
class Study:
    name: str
    description: str
    files: List[str]


# -------- Weights & model --------
def build_model(weight_path: str) -> SyntaxLightningModule:
    m = SyntaxLightningModule(
        num_classes=CFG.num_classes,
        lr=1e-5,
        variant=CFG.variant,
        pl_weight_path=weight_path,
        rnn_hidden_div=CFG.a_rnn.hidden_div,
        rnn_dropout=CFG.a_rnn.dropout,
        bert_nhead=CFG.bert.nhead,
        bert_layers=CFG.bert.num_layers,
        bert_ff_div=CFG.bert.ff_div,
        bert_dropout=CFG.bert.dropout,
        precision=CFG.precision,
    )
    m = m.to(DEVICE).eval()
    return m


def list_weights() -> Tuple[List[str], List[str]]:
    """
    Приоритет:
      1) CFG.weights.left/right
      2) weights/left|right
      3) HF repo (ENV WEIGHTS_REPO или CFG.weights_repo)
    """
    left = list(getattr(CFG.weights, "left", [])) or []
    right = list(getattr(CFG.weights, "right", [])) or []

    if not left:
        left = discover_weights("weights/left")
    if not right:
        right = discover_weights("weights/right")

    if not left and not right:
        repo_id = os.environ.get("WEIGHTS_REPO", getattr(CFG, "weights_repo", "")) or ""
        if repo_id:
            pulled = fetch_weights(repo_id)
            left, right = pulled["left"], pulled["right"]

    return left, right


# -------- Глобальные модели, инициализируемые при импорте --------
_LEFT_MODEL_PATHS, _RIGHT_MODEL_PATHS = list_weights()
_LEFT_MODELS: List[SyntaxLightningModule] = []
_RIGHT_MODELS: List[SyntaxLightningModule] = []

for wp in _LEFT_MODEL_PATHS:
    try:
        _LEFT_MODELS.append(build_model(wp))
    except Exception as e:
        print(f"[WARN] failed to init LEFT model {wp}: {e}")

for wp in _RIGHT_MODEL_PATHS:
    try:
        _RIGHT_MODELS.append(build_model(wp))
    except Exception as e:
        print(f"[WARN] failed to init RIGHT model {wp}: {e}")

if not _LEFT_MODELS and not _RIGHT_MODELS:
    print("[ERROR] No models initialized: check weights/ and HF repo configuration.")


# -------- Stable sort --------
def _stable_sort_paths(paths: List[str]) -> List[str]:
    """
    Сортировка в духе старого пайплайна:
      (SeriesInstanceUID, basename, fullpath)
    """
    def key(p: str):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
            series = str(getattr(ds, "SeriesInstanceUID", ""))
        except Exception:
            series = ""
        return (series, os.path.basename(p), p)
    return sorted(paths, key=key)


def _filter_dicom_paths(paths: List[str]) -> List[str]:
    """Защитная фильтрация: оставляем только корректные DICOM."""
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            pydicom.dcmread(p, stop_before_pixels=True)
            out.append(p)
        except Exception:
            continue
    return out


# -------- Packing one side --------
def _pack_study_side_to_tensor(
    file_paths: List[str],
    frames_per_clip: int,
    video_size: Tuple[int, int],
) -> torch.Tensor | None:
    if not file_paths:
        return None
    tx = test_like_transform(video_size)
    clips = []
    for p in file_paths:
        arr = read_dicom_uint8(p)
        arr = ensure_length_center_crop(arr, frames_per_clip)
        thwc = np.stack([arr, arr, arr], axis=-1)  # (T,H,W,3)
        cthw = tx(torch.tensor(thwc, dtype=torch.uint8))
        clips.append(cthw)
    if not clips:
        return None
    return torch.stack(clips, dim=0).unsqueeze(0)  # (1,S,C,T,H,W)


@torch.no_grad()
def _score_side_by_models(
    side_paths: List[str],
    models: List[SyntaxLightningModule],
    model_paths: List[str],
    frames_per_clip: int,
    video_size: Tuple[int, int],
) -> Dict[str, Any]:
    if not side_paths:
        return {"mean": 0.0, "per_model": [], "n_files": 0}

    x = _pack_study_side_to_tensor(side_paths, frames_per_clip, video_size)
    if x is None:
        return {"mean": 0.0, "per_model": [], "n_files": 0}
    x = x.to(DEVICE)

    per_model_scores: List[float] = []
    used_models: List[str] = []

    for m, wp in zip(models, model_paths):
        try:
            y = m(x)  # (1,2)
            reg_log = float(y[0, 1].detach().cpu().numpy())
            score = float(max(0.0, np.exp(reg_log) - 1.0))  # inverse log(1+score)
            per_model_scores.append(score)
            used_models.append(os.path.basename(wp))
        except Exception as e:
            print(f"[WARN] model {wp} failed: {e}")

    mean_score = float(np.mean(per_model_scores)) if per_model_scores else 0.0
    return {
        "mean": mean_score,
        "per_model": [{"model": n, "score": round(s, 3)} for n, s in zip(used_models, per_model_scores)],
        "n_files": len(side_paths),
    }


# -------- Top-level inference --------
def run_inference(studies: List[Study]) -> Dict[str, Any]:
    if not _LEFT_MODEL_PATHS and not _RIGHT_MODEL_PATHS:
        return {"error": "No weights found. Upload to weights/left and weights/right."}
    if not _LEFT_MODELS and not _RIGHT_MODELS:
        return {"error": "Models failed to initialize. See logs."}

    results = {"studies": []}
    thr = CFG.thresholds.both
    video_size = tuple(CFG.video_size)
    frames = CFG.frames_per_clip

    for st in studies:
        st_files = _filter_dicom_paths(st.files)

        cls = classify_artery(st_files)  # {"left":[{path,prob}], "right":[...], "other":[...]}

        left_paths = _stable_sort_paths([r["path"] for r in cls["left"]])
        right_paths = _stable_sort_paths([r["path"] for r in cls["right"]])

        left_res = _score_side_by_models(
            left_paths,
            _LEFT_MODELS,
            _LEFT_MODEL_PATHS,
            frames,
            video_size,
        )
        right_res = _score_side_by_models(
            right_paths,
            _RIGHT_MODELS,
            _RIGHT_MODEL_PATHS,
            frames,
            video_size,
        )

        total = left_res["mean"] + right_res["mean"]

        results["studies"].append({
            "study": st.name,
            "description": st.description or "",
            "left": {
                "mean": round(left_res["mean"], 3),
                "per_model": left_res["per_model"],
                "n_files": left_res["n_files"],
                "files": cls["left"],
            },
            "right": {
                "mean": round(right_res["mean"], 3),
                "per_model": right_res["per_model"],
                "n_files": right_res["n_files"],
                "files": cls["right"],
            },
            "other": cls["other"],
            "total": {
                "mean": round(total, 3),
                f"High-risk (≥{thr:.1f})": bool(total >= thr),
            },
        })
    return results
