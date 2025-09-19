# schemas.py
# 스키마 관리

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"

@dataclass(frozen=True, slots=True)
class DatasetSchema:
    name: str
    target_name: str
    task_type: Optional[TaskType] = None # None이면 전처리에서 자동 감지
    numeric_features: Optional[Sequence[str]] = None
    categorical_features: Optional[Sequence[str]] = None
    drop_features: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self):
        if self.numeric_features and self.categorical_features:
            inter = set(self.numeric_features).intersection(self.categorical_features)
            if inter:
                raise ValueError(f"[{self.name}] numeric & categorical overlap: {sorted(inter)}")
        if self.target_name in set(self.drop_features):
            raise ValueError(f"[{self.name}] target_name is in drop_features: {self.target_name}")