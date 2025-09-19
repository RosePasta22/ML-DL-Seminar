from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Iterable

# noise_types에서 모든 변환 로직 import
from .noise_types import (
    NoiseConfig,
    make_pairflip_T,
    make_classdep_T,
    apply_label_noise,
    apply_feature_noise_gaussian,
    apply_feature_noise_spike,
)

# =============================
# Train split 전용 오케스트레이터
# =============================
def apply_noise_to_train_split(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    num_classes: int,
    config: NoiseConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """train split에만 노이즈 적용"""
    X_tr2, y_tr2, meta = _apply_noise_one_split(X_tr, y_tr, num_classes, config)
    return X_tr2, y_tr2, meta

# =============================
# Multi-split 오케스트레이터
# =============================
def apply_noise_to_selected_splits(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    num_classes: int,
    config: NoiseConfig,
    targets: Iterable[str] = ("train",),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """train/val/test 중 지정된 split에 노이즈 적용"""
    targets = set(t.lower() for t in targets)
    X_tr2, y_tr2, X_val2, y_val2, X_te2, y_te2 = X_tr, y_tr, X_val, y_val, X_te, y_te
    meta: Dict[str, Any] = {"train": None, "val": None, "test": None}

    if "train" in targets:
        X_tr2, y_tr2, meta["train"] = _apply_noise_one_split(X_tr2, y_tr2, num_classes, config)
    if "val" in targets:
        X_val2, y_val2, meta["val"] = _apply_noise_one_split(X_val2, y_val2, num_classes, config)
    if "test" in targets:
        X_te2, y_te2, meta["test"] = _apply_noise_one_split(X_te2, y_te2, num_classes, config)

    return X_tr2, y_tr2, X_val2, y_val2, X_te2, y_te2, meta

# =============================
# 단일 split 노이즈 적용
# =============================
def _apply_noise_one_split(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    config: NoiseConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """하나의 split에 노이즈 적용"""
    meta: Dict[str, Any] = {"kind": config.kind}
    X_out, y_out = X, y

    # --- Feature noise ---
    if config.is_feature_enabled():
        meta["feature_mode"] = config.feature_mode
        meta["seed_feature"] = config.seed_feature
        if config.feature_mode == "gaussian":
            X_out, idx_feat = apply_feature_noise_gaussian(
                X_out, frac=config.feature_frac, scale=config.feature_scale,
                seed_noise=int(config.seed_feature or 0)
            )
        elif config.feature_mode == "spike":
            X_out, idx_feat = apply_feature_noise_spike(
                X_out, frac=config.spike_frac, spike_value=config.spike_value,
                seed_noise=int(config.seed_feature or 0)
            )
        else:
            raise ValueError(f"unknown feature mode: {config.feature_mode}")
        meta["feature_idx"] = idx_feat

    # --- Label noise ---
    if config.is_label_enabled():
        meta["label_mode"] = config.label_mode
        meta["label_rate"] = config.label_rate
        meta["seed_label"] = config.seed_label

        T = None; transition = None
        if config.label_mode == "pairflip":
            transition = make_pairflip_T(num_classes, config.label_rate, config.pairflip_pairs)
        elif config.label_mode == "classdep":
            transition = make_classdep_T(num_classes, config.classdep_etas)

        y_out, idx_lbl, y_orig, T = apply_label_noise(
            y_out, num_classes=num_classes, mode=config.label_mode,
            noise_rate=config.label_rate, seed_noise=int(config.seed_label or 0),
            transition=transition, X=X_out if config.label_mode == "instancedep" else None,
            tau=config.instancedep_tau,
        )
        meta["label_idx"] = idx_lbl
        meta["label_orig"] = y_orig
        meta["transition"] = T

    return X_out, y_out, meta
