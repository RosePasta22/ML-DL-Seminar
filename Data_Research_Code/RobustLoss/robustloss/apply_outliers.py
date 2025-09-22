# apply_outliers.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Any, Dict, List

from .schemas import DatasetSchema
from .outliers import OutlierConfig, make_outliers


# =========================
# Public API
# =========================
def apply_outliers(
    df: pd.DataFrame,                 # 대상 DF (train/val/test 중 하나)
    schema: DatasetSchema,            # DatasetSchema 객체 (target_name 포함)
    config: OutlierConfig,            # outlier 생성 설정
    *,
    cols: Optional[Sequence[str]] = None,  # 변조할 feature 목록 (None이면 자동)
    meta: bool = True,                     # 메타 열 추가 여부
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    단일 DF에 outlier 행을 주입한다.
    - 입력: df (split 단위)
    - 동작: df 기준으로 μ,σ 추정 → df에서 템플릿 샘플 → 일부 feature 변조 → outlier 행 생성
    - 반환:
        df_aug   : 원래 DF + outlier
        out_only : outlier 행만 (메타 열 포함 가능)
    """
    out_only = make_outliers(
        df=df,            # 스키마 기준(열 순서/타입)
        schema=schema,    # target_name 제공
        config=config,    # outlier 설정
        cols=cols,        # 변조 feature 지정(없으면 자동)
        base=df,          # μ,σ/템플릿 기준 = 현재 df
        meta=meta,        # 메타 열 추가 여부
    )
    df_aug = pd.concat([df, out_only], ignore_index=True)
    return df_aug, out_only


# =========================
# Utilities (for pipeline)
# =========================
def _feature_names(schema: DatasetSchema, n_features: int) -> List[str]:
    """
    스키마에서 feature 이름을 얻되 없으면 f0..f{d-1} 생성.
    - schema.feature_names가 존재하고 길이가 맞으면 그대로 사용
    - 아니면 f0, f1, ... 형태로 자동 생성
    """
    if getattr(schema, "feature_names", None) and len(schema.feature_names) == n_features:
        return list(schema.feature_names)
    return [f"f{i}" for i in range(n_features)]


def _xy_to_df(X: np.ndarray, y: np.ndarray, schema: DatasetSchema) -> pd.DataFrame:
    """
    (X, y) → DataFrame(+타깃) 변환.
    - 열 이름은 스키마에서 가져오거나 자동 생성
    - 마지막에 target 열(schema.target_name) 추가
    """
    cols = _feature_names(schema, X.shape[1])
    df = pd.DataFrame(X, columns=cols)
    df[schema.target_name] = y
    return df


def _df_to_xy(df: pd.DataFrame, schema: DatasetSchema) -> Tuple[np.ndarray, np.ndarray]:
    """
    DataFrame(+타깃) → (X, y) 변환.
    - outlier 메타 열은 학습 입력에 섞이면 안 되므로 자동 드롭
    - 드롭 대상: "_is_outlier", "_cols_outlier", "_zscore_outlier", "_zrange"
    """

    drop_meta_candidates = [
    "_is_outlier", "_cols_outlier", "_zscore_outlier", "_zrange",
    "_m_changed", "_z_avg", "_z_std", "_z_min", "_z_max",
    "_m_avg", "_m_min", "_m_max"
    ]
    drop_meta = [c for c in drop_meta_candidates if c in df.columns]

    if drop_meta:
        df = df.drop(columns=drop_meta)
    X = df.drop(columns=[schema.target_name]).to_numpy()
    y = df[schema.target_name].to_numpy()
    return X, y


def _summarize_outliers(out_only_df: pd.DataFrame, config: "OutlierConfig") -> Dict[str, Any] | None:
    """
    outlier 행 요약치(개수, m 통계, z 통계 등)를 dict로 반환.
    - 입력: out_only_df (apply_outliers → 반환된 outlier 전용 DF)
    - 출력 예:
        {
          "n_added": 120,
          "rate": 0.2,
          "m_avg": 2.7,
          "m_min": 1,
          "m_max": 5,
          "z_avg": 3.95,
          "z_std": 0.52,
          "z_min": 3.01,
          "z_max": 4.98,
          "zmin": 3.0,
          "zmax": 5.0,
          "two_side": True,
          "seed_outlier": 42
        }
    """
    if out_only_df is None or len(out_only_df) == 0:
        return None

    n_added = int(len(out_only_df))

    # outliers.py에서 이미 붙여둔 summary 메타 열 활용
    z_avg = float(out_only_df["_z_avg"].iloc[0]) if "_z_avg" in out_only_df else None
    z_std = float(out_only_df["_z_std"].iloc[0]) if "_z_std" in out_only_df else None
    z_min = float(out_only_df["_z_min"].iloc[0]) if "_z_min" in out_only_df else None
    z_max = float(out_only_df["_z_max"].iloc[0]) if "_z_max" in out_only_df else None

    m_avg = float(out_only_df["_m_avg"].iloc[0]) if "_m_avg" in out_only_df else None
    m_min = int(out_only_df["_m_min"].iloc[0]) if "_m_min" in out_only_df else None
    m_max = int(out_only_df["_m_max"].iloc[0]) if "_m_max" in out_only_df else None

    return {
        "n_added": n_added,
        "rate": float(config.rate),
        "m_avg": m_avg,
        "m_min": m_min,
        "m_max": m_max,
        "z_avg": z_avg,
        "z_std": z_std,
        "z_min": z_min,
        "z_max": z_max,
        "zmin": float(config.zmin),
        "zmax": float(config.zmax),
        "two_side": bool(config.two_side),
        "seed_outlier": (None if config.seed_outlier is None else int(config.seed_outlier)),
    }
