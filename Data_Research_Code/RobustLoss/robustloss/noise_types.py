from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Tuple

# =============================
# Noise Types & Configuration
# =============================

# 라벨 노이즈 종류
LabelMode = Literal["symmetric", "pairflip", "classdep", "instancedep"]
# 피처 노이즈 종류
FeatureMode = Literal["gaussian", "spike"]

@dataclass(slots=True)
class NoiseConfig:
    """노이즈 적용을 위한 설정 클래스.
    kind: 어떤 타입의 노이즈를 적용할지 지정 ("none","label","feature","both")
    """

    kind: Literal["none", "label", "feature", "both"] = "none"

    # --- Label Noise 설정 ---
    label_mode: Optional[LabelMode] = None     # 노이즈 종류
    label_rate: float = 0.0                    # 노이즈율 η
    seed_label: Optional[int] = None           # 라벨 노이즈 랜덤 시드
    pairflip_pairs: Optional[Dict[int, int]] = None  # pairflip용 클래스 쌍
    classdep_etas: Optional[np.ndarray] = None       # class-dependent 노이즈율 벡터
    instancedep_tau: float = 1.0                     # instance-dependent scaling factor

    # --- Feature Noise 설정 ---
    feature_mode: Optional[FeatureMode] = None # 피처 노이즈 방식
    seed_feature: Optional[int] = None         # 피처 노이즈 랜덤 시드
    feature_frac: float = 0.0                  # 전체 샘플 중 노이즈 적용 비율
    feature_scale: float = 0.0                 # Gaussian scale (std 비율)
    spike_frac: float = 0.0                    # Spike 적용 비율
    spike_value: float = 10.0                  # Spike 값 (outlier 크기)

    # --- Helper methods ---
    def is_label_enabled(self) -> bool:
        """라벨 노이즈를 적용할지 여부"""
        return self.kind in ("label", "both") and self.label_mode is not None and self.label_rate > 0

    def is_feature_enabled(self) -> bool:
        """피처 노이즈를 적용할지 여부"""
        if self.kind not in ("feature", "both"):
            return False
        if self.feature_mode is None:
            return False
        if self.feature_mode == "gaussian":
            return self.feature_frac > 0 and self.feature_scale > 0
        if self.feature_mode == "spike":
            return self.spike_frac > 0
        return False


# =============================
# Label Noise Functions
# =============================

def _categorical_sample(T_row: np.ndarray, rng: np.random.Generator) -> int:
    """전이행렬(T_row) 기반으로 새로운 라벨을 샘플링"""
    cdf = np.cumsum(T_row)
    r = rng.random()
    return int(np.searchsorted(cdf, r, side="right"))

def make_pairflip_T(num_classes: int, eta: float, pairs: Dict[int, int]) -> np.ndarray:
    """Pairflip 전이행렬 생성"""
    T = np.eye(num_classes) * (1.0 - eta)
    for i, j in pairs.items():
        T[i, j] = eta
    return T

def make_classdep_T(num_classes: int, etas: np.ndarray) -> np.ndarray:
    """Class-dependent 전이행렬 생성"""
    assert etas.shape == (num_classes,)
    T = np.zeros((num_classes, num_classes), dtype=float)
    for k in range(num_classes):
        eta = float(etas[k])
        T[k, k] = 1.0 - eta
        if num_classes > 1:
            rest = eta / (num_classes - 1)
            for j in range(num_classes):
                if j != k:
                    T[k, j] = rest
    return T

def apply_label_noise(
    y: np.ndarray,
    num_classes: int,
    mode: LabelMode = "symmetric",
    noise_rate: float = 0.2,
    seed_noise: int = 123,
    transition: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """라벨 노이즈 적용
    반환: (노이즈 라벨, 변경된 인덱스, 변경 전 라벨, 전이행렬)
    """
    assert y.ndim == 1
    K = int(num_classes)
    y = y.astype(int, copy=True)
    rng = np.random.default_rng(seed_noise)

    # Instance-dependent 노이즈
    if mode == "instancedep":
        assert X is not None, "Instance-dependent noise needs X"
        D = X.shape[1]
        W = rng.normal(0, 1, size=(K, D))
        b = rng.normal(0, 1, size=(K,))
        logits = (X @ W.T + b) / max(1e-8, float(tau))
        mask_true = np.eye(K, dtype=bool)[y]
        logits_masked = np.where(mask_true, -1e9, logits)
        max_log = logits_masked.max(axis=1, keepdims=True)
        exps = np.exp(logits_masked - max_log)
        q = exps / exps.sum(axis=1, keepdims=True)
        y_noisy = y.copy()
        flip = rng.random(len(y)) < noise_rate
        idx = np.where(flip)[0]
        if len(idx) > 0:
            choices = [rng.choice(K, p=q[i]) for i in idx]
            y_noisy[idx] = np.array(choices, dtype=int)
        idx_changed = np.where(y_noisy != y)[0]
        return y_noisy, idx_changed, y[idx_changed], None

    # Symmetric / Pairflip / Classdep
    if mode == "symmetric":
        eta = float(noise_rate)
        T = np.full((K, K), eta / (K - 1)) if K > 1 else np.array([[1.0]])
        np.fill_diagonal(T, 1.0 - eta)
    elif mode in ("pairflip", "classdep"):
        assert transition is not None, f"{mode} requires transition matrix"
        T = transition
    else:
        raise ValueError(f"unknown label mode: {mode}")

    y_noisy = y.copy()
    for i in range(len(y)):
        y_noisy[i] = _categorical_sample(T[y[i]], rng)
    idx_changed = np.where(y_noisy != y)[0]
    return y_noisy, idx_changed, y[idx_changed], T


# =============================
# Feature Noise Functions
# =============================

def apply_feature_noise_gaussian(
    X: np.ndarray,
    frac: float = 0.1,
    scale: float = 0.5,
    seed_noise: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian 노이즈 추가: 일부 샘플에 feature-wise std 기반으로 가우시안 노이즈 주입"""
    rng = np.random.default_rng(seed_noise)
    N, D = X.shape
    m = int(round(N * frac))
    if m == 0:
        return X.copy(), np.empty((0,), dtype=int)
    idx = rng.choice(N, size=m, replace=False)
    std = X.std(axis=0, ddof=1)
    noise = rng.normal(0.0, 1.0, size=(m, D)) * (std * scale)
    X_out = X.copy(); X_out[idx] = X_out[idx] + noise
    return X_out, idx

def apply_feature_noise_spike(
    X: np.ndarray,
    frac: float = 0.05,
    spike_value: float = 10.0,
    seed_noise: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spike 노이즈 추가: 일부 샘플을 spike_value로 치환 (극단적 outlier 생성)"""
    rng = np.random.default_rng(seed_noise)
    N, D = X.shape
    m = int(round(N * frac))
    if m == 0:
        return X.copy(), np.empty((0,), dtype=int)
    idx = rng.choice(N, size=m, replace=False)
    X_out = X.copy(); X_out[idx] = spike_value
    return X_out, idx
