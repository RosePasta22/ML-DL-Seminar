# robustloss-lab

Robust classification loss toolkit (CCE/SCCE, Focal, GCE) with simple experiment runners for noisy labels and outliers.  
노이즈/아웃라이어 환경에서 강건한 분류 실험을 위한 손실 함수 라이브러리와 실험 러너를 제공합니다.

---

## ✨ Features
- Losses: CE, GCE, Focal, **CCE**, **SCCE**
- Experiment runners: `run_experiment`, `run_clean_vs_noise`, `run_clean_vs_outlier`
- Noise / Outlier injection utilities and metadata logging
- Minimal, sklearn-like workflow with PyTorch backend

---

## 📦 Install

> **Note**: PyTorch는 CUDA/CPU 환경에 맞춰 별도 설치하세요.  
> 예:  
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

```bash
pip install robustloss-lab
```

---

## 🚀 Quick start

```python
from robustloss import (
    DatasetSchema, TaskType,
    make_loss, run_experiment, plot_history,
    NoiseConfig, OutlierConfig, pct_drop
)

# 1) Define dataset schema
schema = DatasetSchema(
    name="uci_wine",
    target_name="class",
    task_type=TaskType.MULTICLASS
)

# 2) Choose a loss (e.g., SCCE)
loss_fn = make_loss("scce", eps=1e-3)

# 3) Run a quick experiment
model, hist, report = run_experiment(
    df, schema,
    loss_fn=loss_fn, loss_name="SCCE",
    epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4,
    optimizer_name="adam", seed=42
)

print(report)  # dict(test_acc=..., test_f1=..., noise_meta=..., outlier_meta=...)
plot_history([hist], ["SCCE"])
```

---

## 📂 Modules

**Core (required submodules)**  
- `schemas.py` — 데이터 스키마 정의  
- `preprocess.py`, `datamod.py` — 전처리 / split  
- `loss_functions.py` — CE, GCE, Focal, CCE, SCCE  
- `models.py` — Logistic/Softmax linear classifiers  
- `train_many.py` — 학습 루프, early stopping, 시각화  

**Optional submodules**  
- `noise_types.py`, `apply_noise.py` — 라벨/피처 노이즈  
- `outliers.py`, `apply_outliers.py` — 아웃라이어 생성/주입  

> 선택 모듈은 환경에 따라 미포함일 수 있으며, 패키지 import 시 `None`으로 바인딩될 수 있습니다.

---

## 🧩 API Sketch

### `DatasetSchema`
```python
@dataclass(frozen=True, slots=True)
class DatasetSchema:
    name: str
    target_name: str
    task_type: Optional[TaskType] = None
    numeric_features: Optional[Sequence[str]] = None
    categorical_features: Optional[Sequence[str]] = None
    drop_features: Sequence[str] = field(default_factory=tuple)
```

### `NoiseConfig`
```python
@dataclass(slots=True)
class NoiseConfig:
    kind: Literal["none", "label", "feature", "both"] = "none"
    label_mode: Optional[LabelMode] = None
    label_rate: float = 0.0
    seed_label: Optional[int] = None
    feature_mode: Optional[FeatureMode] = None
    seed_feature: Optional[int] = None
    feature_frac: float = 0.0
    feature_scale: float = 0.0
    spike_frac: float = 0.0
    spike_value: float = 10.0
```

### `OutlierConfig`
```python
@dataclass(frozen=True, slots=True)
class OutlierConfig:
    spike_value: float = 10.0
    rate: float = 0.1
    zmin: float = 3.0
    zmax: float = 5.0
    mmin: int = 1
    mmax: Optional[int] = None
    two_side: bool = True
    seed_outlier: Optional[int] = 42
    target: Iterable[str] = ("train",)
```

### Experiment Runners
```python
model, hist, report = run_experiment(df, schema, loss_fn, ...)

[h_c, h_n], labels, df_res = run_clean_vs_noise(df, schema, loss_fn=loss_fn, ...)

[h_c, h_o], labels, df_res = run_clean_vs_outlier(df, schema, loss_fn=loss_fn, ...)
```

### Metadata
- `noise_meta`: 라벨/피처 노이즈 적용 정보 (전이행렬, 인덱스, 시드 등)  
- `outlier_meta`: 아웃라이어 주입 요약 (비율, |z|-통계, m범위/시드 등)

---

## 📝 Patch Notes
- 1.0.0: First release  
- 2.0.x: 라이브러리 모듈화, 노이즈 기능 추가  
- 2.1.0: Outliers 모듈 추가  
- 2.1.4: SCCE clamp 재정의(q_t)  
- **2.1.5: PyPI 패키징 정리, README 개선, 배포명 `robustloss-lab`, import `robustloss`**

---

## 📎 Links
- [Latest Release](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.5)  
- [Homepage / Source](https://github.com/RosePasta22/ML-DL-Seminar)
