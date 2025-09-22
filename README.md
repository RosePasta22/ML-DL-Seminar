# Data Research Code

## Description

데이터 분석에 필요한 스키마 지정, 전처리, 함수 구현, 노이즈 관리, 학습 라이브러리입니다.

## 내부 파일

**1. 데이터 스키마 관리**

 * schemas.py

**2. 데이터 전처리 및 split**

* datamod.py
* preprocess.py

**3. 다양한 loss function 구현**

* loss_functions.py

**4. logistic/softmax 기반 linear classifier**

* models.py

**5. label/feature noise 관리**

* noise_types.py
* apply_noise.py

**6. Outlier 관리**

* outliers.py
* apply_outlier.py
  
**7. 학습 루프, optimizer, early stopping, 시각화**

* train_many.py

  
# **How to use**

## install

[Latest Release](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.2)
```python
pip install "git+https://github.com/RosePasta22/ML-DL-Seminar@v2.1.2#subdirectory=Data_Research_Code/RobustLoss"
```
```python
pip install "git+https://github.com/RosePasta22/ML-DL-Seminar@v2.1.2#subdirectory=Data_Research_Code/RobustLoss"
```
```python
pip install "git+https://github.com/RosePasta22/ML-DL-Seminar@main#subdirectory=Data_Research_Code/RobustLoss"
```
```python
pip install "https://github.com/RosePasta22/ML-DL-Seminar/releases/download/v2.1.2/robustloss-2.1.2-py3-none-any.whl"
```

## import
```python
from robustloss import DatasetSchema, TaskType
from robustloss import make_loss
from robustloss import run_experiment, plot_history, run_clean_vs_noise, pct_drop
from robustloss import NoiseConfig
```

# **Important prototypes**

# **Setting**

## **DatasetSchema**
```python
DatasetSchema(
    name: str
    target_name: str
    task_type: Optional[TaskType] = None       # Task_Type.BINARY / Task_Type.MULTICLASS  None일 시 전처리 시 자동감지 
    numeric_features: Optional[Sequence[str]] = None
    categorical_features: Optional[Sequence[str]] = None
    drop_features: Sequence[str] = field(default_factory=tuple)
)
```

### **예시**
```python
schema = DatasetSchema(
    name="uci_wine",
    target_name="class",
    task_type=TaskType.MULTICLASS
)
```

## **NoiseConfig**
```python

NoiseConfig:
    kind: Literal["none", "label", "feature", "both"] = "none"

    # --- Label Noise ---
    label_mode: Optional[LabelMode] = None     # 노이즈 종류 ["symmetric", "pairflip", "classdep", "instancedep"]
    label_rate: float = 0.0                    # 노이즈율 η
    seed_label: Optional[int] = None           # 라벨 노이즈 랜덤 시드
    pairflip_pairs: Optional[Dict[int, int]] = None  # pairflip용 클래스 쌍
    classdep_etas: Optional[np.ndarray] = None       # class-dependent 노이즈율 벡터
    instancedep_tau: float = 1.0                     # instance-dependent scaling factor

    # --- Feature Noise ---
    feature_mode: Optional[FeatureMode] = None # 노이즈 종류 ["gaussian", "spike"]
    seed_feature: Optional[int] = None         # 피처 노이즈 랜덤 시드
    feature_frac: float = 0.0                  # 전체 샘플 중 노이즈 적용 비율
    feature_scale: float = 0.0                 # Gaussian scale (std 비율)
    spike_frac: float = 0.0                    # Spike 적용 비율
    spike_value: float = 10.0                  # Spike 값 (outlier 크기)
```

## **OutlierConfig**

```python
OutlierConfig:
    spike_value: float = 10.0                  #
    rate: float = 0.1                          # outlier 비율 (0.1=10%)
    zmin: float = 3.0                          # z-score 하한 (3σ)
    zmax: float = 5.0                          # z-score 상한 (5σ)
    mmin: int = 1                              # 한 행에서 변조할 feature 최소 개수
    mmax: Optional[int] = None                 # 한 행에서 변조할 feature 최대 개수 (None=전체)
    two_side: bool = True                      # True: ±, False: +만
    seed_outlier: Optional[int] = 42           # Outlier 시드
    target: Iterable[str] = ("train",)         # 주입할 split ("train","val","test")
```

## **run_experiment**

```python
run_experiment(
    df,                                        # Dataframe
    schema_or_name: Union[str, DatasetSchema], # DatasetSchema 객체 또는 registry.py 의 str
    loss_fn,                       # 사용할 손실 함수 (예: CE, GCE, CCE 등)

    # -------------------------
    # 학습 하이퍼파라미터 (기본 프리셋)
    # -------------------------
    epochs: int = 50,              # 최대 학습 epoch 수
    batch_size: int = 64,          # 미니배치 크기
    lr: float = 1e-3,              # 학습률 (learning rate)
    weight_decay: float = 1e-4,    # L2 정규화 강도 (weight decay)

    optimizer_name: str = "adam",  # 옵티마이저 종류 ("adam" | "sgd" | "sgd_momentum")
    loss_name: str = "loss",       # 손실 함수 이름 (로그 출력/플롯 라벨링용)
    patience: int = 10,            # Early Stopping patience (val_loss 개선 없을 시 중단)

    # -------------------------
    # 실행 환경
    # -------------------------
    seed: int = 42,                # 랜덤 시드 (재현성 보장)
    device: str | None = None,     # 연산 장치 지정 ("cuda", "cpu", None이면 자동)

    # -------------------------
    # noise setting
    # -------------------------
    noise: Optional[NoiseConfig] = None,        # 노이즈 구성 객체 (label/feature 종류, 비율, 시드 등)
    noise_targets: Iterable[str] = ("train",),  # 노이즈 적용 대상 split ("train","val","test" 중 선택)
):

return model, hist, dict(test_acc=test_acc, test_f1=test_f1, noise_meta=noise_meta, outlier_meta=outlier_meta)

```

## **run_clean_vs_noise**

```python
run_clean_vs_noise(
    df,                            # Dataframe
    schema_or_name,                # DatasetSchema 객체 또는 registry.py 의 str
    *,
    loss_fn,                       # 사용할 손실 함수 (예: CE, GCE, CCE 등)
    loss_name: str = "loss",       # 손실 함수 이름 (로그 출력/플롯 라벨링용)
    seed: int = 42,                # 랜덤 시드 (재현성 보장)

    # -------------------------
    # 학습 하이퍼파라미터 (기본 프리셋)
    # -------------------------
    epochs: int = 50,              # 최대 학습 epoch 수
    batch_size: int = 64,          # 미니배치 크기
    lr: float = 1e-3,              # 학습률 (learning rate)
    weight_decay: float = 1e-4,    # L2 정규화 강도 (weight decay)

    optimizer_name: str = "adam",  # 옵티마이저 종류 ("adam" | "sgd" | "sgd_momentum")
    patience: int = 10,            # Early Stopping patience (val_loss 개선 없을 시 중단)
    device: str | None = None,

    # -------------------------
    # noise setting
    # -------------------------
    noise_cfg: Optional["NoiseConfig"] = None,
    noise_targets: Iterable[str] = ("train",),
):

return ( [hist_c, hist_n], ["CLEAN", "NOISE"], df_results )

```

## **run_clean_vs_outlier

```python
run_clean_vs_outlier(
    df,                               # 전체 데이터셋 (pandas DataFrame)
    schema_or_name,                   # DatasetSchema 객체 또는 str (등록된 스키마 이름)

    *,
    loss_fn,                          # 사용할 손실 함수 (callable, 예: ce_loss, gce_loss 등)
    loss_name: str = "loss",          # 손실 함수 이름 (로그/라벨링용 표시)

    seed: int = 42,                   # 랜덤 시드 (데이터 분할/학습 재현성 보장)

    # -------------------------
    # 학습 하이퍼파라미터
    # -------------------------
    epochs: int = 50,                 # 최대 학습 epoch 수
    batch_size: int = 64,             # 미니배치 크기
    lr: float = 1e-3,                 # 학습률 (learning rate)
    weight_decay: float = 1e-4,       # L2 정규화 강도 (weight decay)

    optimizer_name: str = "adam",     # 옵티마이저 종류 ("adam" | "sgd" | "sgd_momentum")
    patience: int = 10,               # Early Stopping patience (val_loss 개선 없을 시 중단)

    # -------------------------
    # 실행 환경
    # -------------------------
    device: str | None = None,        # 연산 장치 ("cuda", "cpu", None → 자동 감지)

    # -------------------------
    # 아웃라이어 설정
    # -------------------------
    outlier_cfg: Optional[OutlierConfig] = None,  # OutlierConfig 객체 (rate, z 범위, m 범위 등 설정)
):

return ([hist_c, hist_o], ["CLEAN", "OUTLIER"], df_results)

```

# Patch Note
* 1.0.0 : 최초 릴리즈
* 1.0.1 : 로그 수정
* 2.0.0 : 라이브러리 모듈화, 노이즈 추가 [(패치내역)](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.0.0)
* 2.0.1 : 버그 수정
* 2.0.2 : 버그 수정
* 2.0.3 : 버그 수정
* 2.0.4 : 버그 수정 [(패치내역)](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.0.4)
