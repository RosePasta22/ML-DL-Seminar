# outliers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple, Iterable
import numpy as np
import pandas as pd
from numpy.random import default_rng
from .schemas import DatasetSchema

@dataclass(frozen=True, slots=True)
class OutlierConfig:
    rate: float = 0.1                  # outlier 비율 (0.1=10%)
    zmin: float = 3.0                  # z-score 하한
    zmax: float = 5.0                  # z-score 상한
    mmin: int = 1                      # 한 행에서 변조할 feature 최소 개수
    mmax: Optional[int] = None         # 한 행에서 변조할 feature 최대 개수 (None=전체)
    two_side: bool = True              # True: ±, False: +만
    seed_outlier: Optional[int] = 42           # 난수 시드
    target: Iterable[str] = ("train",) # 주입할 split ("train","val","test")

# -----------------------
# 내부 유틸
# -----------------------
def _num_cols(df: pd.DataFrame, target: str) -> List[str]:
    """타깃 컬럼을 제외한 수치형 feature 목록"""
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != target]

def _mu_sigma(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """각 feature의 평균(μ)과 표준편차(σ)"""
    mu = df[cols].mean(0).to_numpy(float)
    sigma = df[cols].std(0, ddof=0).to_numpy(float)
    return mu, sigma

# -----------------------
# 공개 API
# -----------------------
def make_outliers(
    df: pd.DataFrame,                 # 스키마 기준(열 순서/타입)
    schema: DatasetSchema,            # DatasetSchema (target_name 포함)
    config: OutlierConfig,            # outlier 설정
    *,
    cols: Optional[Sequence[str]] = None,  # 변조 대상 feature 목록 (None=자동)
    base: Optional[pd.DataFrame] = None,   # μ,σ/템플릿 기준 DF (보통 해당 split DF)
    meta: bool = True,                      # 메타 열 추가 여부
) -> pd.DataFrame:
    """
    원본 df는 수정하지 않고 outlier '행'만 생성하여 반환한다.
    """
    target = schema.target_name
    rng = default_rng(config.seed_outlier)
    base_df = base if base is not None else df

    # 1) 변조 대상 feature
    if cols is None:
        cols = _num_cols(base_df, target)
    cols = list(cols)
    if not cols:
        raise ValueError("수치형 feature 없음")

    # 2) μ,σ (σ=0 제외)
    mu, sigma = _mu_sigma(base_df, cols)
    valid = sigma > 0
    cols = [c for c, ok in zip(cols, valid) if ok]
    mu = mu[valid]; sigma = sigma[valid]
    if not cols:
        raise ValueError("표준편차 0인 feature만 존재하여 변조 불가")

    # 3) 생성 개수
    n = len(base_df)                         # 기준 DF 행 수
    k = int(np.ceil(config.rate * n))        # 생성할 outlier 행 수
    if k == 0:
        cols_all = list(df.columns) + (["_is_outlier", "_cols_outlier", "_zscore_outlier", "_zrange"] if meta else [])
        return pd.DataFrame(columns=cols_all)

    # 4) 한 행당 변조 feature 수
    mmax = config.mmax if config.mmax is not None else len(cols)
    mmax = int(min(max(mmax, config.mmin), len(cols)))
    mmin = int(max(1, min(config.mmin, mmax)))

    # 5) 템플릿 행 샘플링
    idx = rng.integers(low=0, high=n, size=k, endpoint=False)
    out = base_df.iloc[idx].copy(deep=True)

    # 메타 버퍼
    meta_cols_list: List[str] = []
    meta_zscore_list: List[str] = []
    meta_zrange_list: List[str] = []

    # 6) 행별 변조
    for i in range(k):
        m = rng.integers(low=mmin, high=mmax + 1)          # 이번 행에서 변조할 feature 개수
        sel = rng.choice(len(cols), size=m, replace=False) # 변조할 feature 인덱스들

        z_abs = rng.uniform(config.zmin, config.zmax, size=m)             # 절대 z
        sign = rng.choice([-1.0, 1.0], size=m) if config.two_side else np.ones(m)  # 부호
        z_signed = sign * z_abs                                           # 부호 포함 z-score
        vals = mu[sel] + z_signed * sigma[sel]                            # x = μ + z * σ

        # 실제 값 치환
        row = out.iloc[i]
        for j, v in zip(sel, vals):
            row[cols[j]] = v
        out.iloc[i] = row

        if meta:
            # 변조된 feature 이름들 (sel 순서 유지)
            cols_changed = ",".join([cols[j] for j in sel])
            # 각 feature에 대응하는 (부호 포함) z-score 목록
            zscore_str = ",".join([f"{z:.3f}" for z in z_signed])
            # 설정 요약(범위+방향)
            zrange_str = f"{config.zmin:.1f}~{config.zmax:.1f}" + (" (±)" if config.two_side else " (+)")

            meta_cols_list.append(cols_changed)
            meta_zscore_list.append(zscore_str)
            meta_zrange_list.append(zrange_str)

    # 7) 반환 스키마 정렬
    out = out.reindex(columns=df.columns, fill_value=np.nan)

    # 8) 메타 열 추가
    if meta:
        out["_is_outlier"]      = True
        out["_cols_outlier"]    = meta_cols_list
        out["_zscore_outlier"]  = meta_zscore_list   # 각 행마다 쉼표로 이어진 per-feature z-score(부호 포함)
        out["_zrange"]          = meta_zrange_list   # 설정 요약(예: "3.0~5.0 (±)")

    return out
