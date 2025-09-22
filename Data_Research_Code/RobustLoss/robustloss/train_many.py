# train_many.py
import random, numpy as np, torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence
from typing import Union

# Preprocess & Model Build
from .registry import REGISTRY
from .datamod import prepare_dataset
from .models import build_model
from .schemas import DatasetSchema

# Noise
from typing import Iterable, Optional, Dict, Any
from .noise_types import NoiseConfig
from .apply_noise import (
    apply_noise_to_train_split,
    apply_noise_to_selected_splits,
)

# Outliers
from .outliers import OutlierConfig
from .apply_outliers import apply_outliers, _xy_to_df, _df_to_xy, _summarize_outliers



# -------------------
# 재현성: 시드 고정
# -------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------
# 유틸
# -------------------
def _to_tensor(X: np.ndarray, y: np.ndarray, device=None, dtype=torch.float32):
    X_t = torch.as_tensor(X, dtype=dtype, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long,  device=device)
    return X_t, y_t

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def predict_labels(model: torch.nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds

# -------------------
# 학습 루프(경사하강법 포함 + EarlyStopping) 유틸
# -------------------
def train(
    model: torch.nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    loss_fn,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adam",    # "adam" | "sgd" | "sgd_momentum"
    patience: int = 10,
    device: str | None = None,
    dtype = torch.float32,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=dtype)

    Xtr_t, ytr_t = _to_tensor(X_train, y_train, device=device, dtype=dtype)
    Xva_t, yva_t = _to_tensor(X_val,   y_val,   device=device, dtype=dtype)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.0  # 오리지널 SGD
        )
    elif optimizer_name.lower() == "sgd_momentum":
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9  # SGD + Momentum
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
        )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    best_val = float("inf")
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t)
            val_loss = loss_fn(val_logits, yva_t).item()
            val_acc  = accuracy_from_logits(val_logits, yva_t)
            y_pred = val_logits.argmax(dim=1).cpu().numpy()
            y_true = yva_t.cpu().numpy()
            val_f1 = f1_score(y_true, y_pred, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"[{ep:03d}/{epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        # Early Stopping
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

# clean 대비 상대적 하락(%)
def pct_drop(a_clean, a_modified_data):
    if a_clean == 0: return 0.0
    return (a_clean - a_modified_data) / a_clean * 100.0

# -------------------
# 그래프 유틸
# -------------------
def plot_history(histories: Sequence[dict], labels: Sequence[str],
                 title_suffix: str = "", outdir: str | Path = "plots",
                 show: bool = True, close: bool = True):
    assert len(histories) == len(labels)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    def _save_show(path):
        plt.savefig(path, dpi=150, bbox_inches="tight")
        if show: plt.show()
        if close: plt.close()

    # Val Acc
    plt.figure(figsize=(7,5))
    for h, lab in zip(histories, labels): plt.plot(h["val_acc"], label=lab)
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy"); plt.title(f"Validation Accuracy {title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend()
    _save_show(outdir / f"val_acc{('_' + title_suffix.replace(' ', '_')) if title_suffix else ''}.png")

    # Val Loss
    plt.figure(figsize=(7,5))
    for h, lab in zip(histories, labels): plt.plot(h["val_loss"], label=lab)
    plt.xlabel("Epoch"); plt.ylabel("Validation Loss"); plt.title(f"Validation Loss {title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend()
    _save_show(outdir / f"val_loss{('_' + title_suffix.replace(' ', '_')) if title_suffix else ''}.png")

    # Train Loss
    plt.figure(figsize=(7,5))
    for h, lab in zip(histories, labels): plt.plot(h["train_loss"], label=lab)
    plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title(f"Train Loss {title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend()
    _save_show(outdir / f"train_loss{('_' + title_suffix.replace(' ', '_')) if title_suffix else ''}.png")

# -------------------
# 하이퍼 추천(옵션)
# -------------------
def suggest_hparams(num_train: int, dim: int | None = None, noisy: bool = False):
    if num_train <= 2_000:    cfg = dict(batch_size=32,  epochs=100, lr=1e-3, patience=15)
    elif num_train <= 20_000: cfg = dict(batch_size=64,  epochs=60,  lr=1e-3, patience=10)
    elif num_train <= 200_000:cfg = dict(batch_size=128, epochs=40,  lr=8e-4, patience=7)
    else:                     cfg = dict(batch_size=256, epochs=20,  lr=5e-4, patience=5)
    if dim is not None and dim > 500:
        cfg["weight_decay"] = 3e-4
    if noisy:
        cfg["epochs"] = int(cfg["epochs"] * 0.8)
    return cfg

# ===================
# Main Experiment Runner
# ===================
# 
def run_experiment(
    df,
    schema_or_name: Union[str, DatasetSchema],
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
    # 노이즈 설정
    # -------------------------
    noise: Optional[NoiseConfig] = None,     # 노이즈 구성 객체 (label/feature 종류, 비율, 시드 등)
    noise_targets: Iterable[str] = ("train",),  # 노이즈 적용 대상 split ("train","val","test" 중 선택)

    # -------------------------
    # outlier 설정
    # -------------------------
    outliers: Optional[OutlierConfig] = None,
):

    set_seed(seed)

    # 이름이면 REGISTRY에서 찾고, 객체면 그대로 사용
    if isinstance(schema_or_name, str):
        from registry import REGISTRY # 필요할 때만 import
        schema = REGISTRY[schema_or_name]
    else:
        schema = schema_or_name
        
    (Xtr, ytr), (Xval, yval), (Xte, yte), meta = prepare_dataset(df, schema, random_state=seed)
    
    # -------------------------
    # 노이즈 적용
    # -------------------------
    K = int(meta["num_classes"])  # 또는 int(np.unique(ytr).size)
    noise_meta: Dict[str, Any] = {"train": None, "val": None, "test": None}
    
    if noise is not None and noise.kind != "none":
        targets = tuple(t.lower() for t in noise_targets)
        if set(targets) == {"train"}:
            # 빠른 경로(호환성 유지): train만 노이즈
            Xtr, ytr, meta_train = apply_noise_to_train_split(
                Xtr, ytr, num_classes=K, config=noise
            )
            noise_meta["train"] = meta_train
        else:
            # 다중 split 적용
            Xtr, ytr, Xval, yval, Xte, yte, meta_all = apply_noise_to_selected_splits(
                Xtr, ytr, Xval, yval, Xte, yte,
                num_classes=K, config=noise, targets=targets
            )
            noise_meta = meta_all
    # -----------------------------------------------

    # -------------------------
    # outliers 적용
    # -------------------------
    outlier_meta: Dict[str, Any] = {"train": None, "val": None, "test": None}
    if outliers is not None:
        targets = set(t.lower() for t in outliers.target)
        allowed = {"train","val","test"}
        invalid = targets - allowed
        if invalid:
            raise ValueError(f"outliers.target 잘못된 값: {invalid} (허용: {allowed})")

        # train
        if "train" in targets:
            df_tr = _xy_to_df(Xtr, ytr, schema)
            df_tr_aug, out_tr = apply_outliers(df_tr, schema, outliers, cols=None, meta=True)
            Xtr, ytr = _df_to_xy(df_tr_aug, schema)
            outlier_meta["train"] = _summarize_outliers(out_tr, outliers)  # ★ z_avg, z_std 포함된 요약 딕셔너리

        # val
        if "val" in targets:
            df_val = _xy_to_df(Xval, yval, schema)
            df_val_aug, out_val = apply_outliers(df_val, schema, outliers, cols=None, meta=True)
            Xval, yval = _df_to_xy(df_val_aug, schema)
            outlier_meta["val"] = _summarize_outliers(out_val, outliers)

        # test
        if "test" in targets:
            df_te = _xy_to_df(Xte, yte, schema)
            df_te_aug, out_te = apply_outliers(df_te, schema, outliers, cols=None, meta=True)
            Xte, yte = _df_to_xy(df_te_aug, schema)
            outlier_meta["test"] = _summarize_outliers(out_te, outliers)
    # -----------------------------------------------

    # 필요 시 자동 추천으로 덮어쓰기 예시:
    # auto = suggest_hparams(num_train=len(Xtr), dim=meta["n_features"])
    # epochs = epochs or auto["epochs"]; batch_size = batch_size or auto["batch_size"]; lr = lr or auto["lr"]; patience = patience or auto["patience"]

    model = build_model(meta["n_features"], meta["num_classes"])

    # 학습 시작 알림
    print(f"{loss_name.upper()} training start...")

    hist = train(
        model, Xtr, ytr, Xval, yval,
        loss_fn=loss_fn,
        epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay,
        optimizer_name=optimizer_name, patience=patience,
        device=device
    )

    # 최종 테스트 평가
    Xte_t, yte_t = _to_tensor(Xte, yte, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        test_acc = accuracy_from_logits(logits, yte_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = yte_t.cpu().numpy()
        test_f1 = f1_score(y_true, y_pred, average="macro")
    
    print(f"\n{loss_name.upper()}\n [TEST] acc={test_acc:.4f}  f1_macro={test_f1:.4f}\n")
    return model, hist, dict(test_acc=test_acc, test_f1=test_f1, noise_meta=noise_meta, outlier_meta=outlier_meta)

# loss에 대해 clean vs noise
def run_clean_vs_noise(
    df,
    schema_or_name,
    *,
    loss_fn,
    loss_name: str = "loss",
    seed: int = 42,
    # train_many.py의 공통 하이퍼들
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adam",
    patience: int = 10,
    device: str | None = None,
    # 노이즈 설정
    noise_cfg: Optional["NoiseConfig"] = None,
    noise_targets: Iterable[str] = ("train",),
):
    """
    같은 분할/시드에서 clean과 noisy(지정한 noise_cfg)를 공정 비교.
    반환: (histories, labels, df_results)
    """

    from .noise_types import NoiseConfig    # 타입 힌트용

    # 1) CLEAN
    model_c, hist_c, score_c = run_experiment(
        df, schema_or_name,
        loss_fn=loss_fn, loss_name=f"{loss_name} (CLEAN)",
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, optimizer_name=optimizer_name,
        patience=patience, seed=seed, device=device,
        noise=None,                       # ← clean
        noise_targets=("train",),         # 무시됨
    )

    # 2) NOISE
    model_n, hist_n, score_n = run_experiment(
        df, schema_or_name,
        loss_fn=loss_fn, loss_name=f"{loss_name} (NOISE)",
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, optimizer_name=optimizer_name,
        patience=patience, seed=seed, device=device,
        noise=noise_cfg,                  # ← 주입
        noise_targets=noise_targets,      # ← 어디에 노이즈를 줄지
    )

    # 3) 집계 표
    rows = []
    def _row(tag, score):
        d = dict(
            setting=tag,
            test_acc=float(score["test_acc"]),
            test_f1=float(score.get("test_f1", float("nan"))),
        )
        # 노이즈 메타(있다면) 일부 표시
        if "noise_meta" in score and isinstance(score["noise_meta"], dict):
            meta = score["noise_meta"]
            # 어디 split이 오염됐는지 간단 요약
            applied = [k for k, v in meta.items() if v is not None]
            d["noise_targets"] = ",".join(applied) if applied else ""
            # 라벨/피처 모드도 요약 (train 우선)
            for split in ("train","val","test"):
                if meta.get(split):
                    d["label_mode"]   = meta[split].get("label_mode")
                    d["label_rate"]   = meta[split].get("label_rate")
                    d["feature_mode"] = meta[split].get("feature_mode")
                    break
        return d

    rows.append(_row("CLEAN", score_c))
    rows.append(_row("NOISE", score_n))
    df_results = pd.DataFrame(rows)

    return ( [hist_c, hist_n], ["CLEAN", "NOISE"], df_results )

# ===================
# loss에 대해 clean vs outlier
# ===================
from typing import Optional, Iterable, Dict, Any
from .outliers import OutlierConfig

def run_clean_vs_outlier(
    df,
    schema_or_name,
    *,
    loss_fn,
    loss_name: str = "loss",
    seed: int = 42,
    # 공통 하이퍼
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adam",
    patience: int = 10,
    device: str | None = None,
    # 아웃라이어 설정
    outlier_cfg: Optional[OutlierConfig] = None,       # 주입할 설정(예: OutlierConfig(...))
):
    """
    같은 분할/시드에서 clean과 outlier(지정한 outlier_cfg)를 공정 비교.
    반환: (histories, labels, df_results)
    """

    # 1) CLEAN
    model_c, hist_c, score_c = run_experiment(
        df, schema_or_name,
        loss_fn=loss_fn, loss_name=f"{loss_name} (CLEAN)",
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, optimizer_name=optimizer_name,
        patience=patience, seed=seed, device=device,
        noise=None,                       # noise 없음
        noise_targets=("train",),         # 무시됨
        outliers=None,                    # ★ outlier 없음 (clean)
    )

    # 2) OUTLIER
    model_o, hist_o, score_o = run_experiment(
        df, schema_or_name,
        loss_fn=loss_fn, loss_name=f"{loss_name} (OUTLIER)",
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, optimizer_name=optimizer_name,
        patience=patience, seed=seed, device=device,
        noise=None,                       # 필요 시 noise 함께 비교하려면 바꿔도 됨
        noise_targets=("train",),         # 필요 시 수정
        outliers=outlier_cfg,             # ★ outlier 주입
    )

    # 3) 집계 표
    def _flatten_outlier_meta(meta: Dict[str, Any] | None) -> Dict[str, Any]:
        """
        outlier_meta( split별 요약 ) → CSV 친화적 평탄화
        키 예: out_train_n_added, out_train_z_avg, out_train_z_std, ...
        """
        d: Dict[str, Any] = {}
        for split in ("train", "val", "test"):
            m = (meta or {}).get(split)
            d[f"out_{split}_n_added"] = (m.get("n_added") if m else 0)
            d[f"out_{split}_rate"]    = (m.get("rate") if m else 0.0)
            d[f"out_{split}_m_avg"]   = (m.get("m_avg") if m else 0.0)
            d[f"out_{split}_m_min"]   = (m.get("m_min") if m else 0)
            d[f"out_{split}_m_max"]   = (m.get("m_max") if m else 0)
            if m:
                # 기존: zmin, zmax, two_side
                d[f"out_{split}_zmin"]     = m.get("zmin")
                d[f"out_{split}_zmax"]     = m.get("zmax")
                d[f"out_{split}_two_side"] = m.get("two_side")

                # ★ 추가: outliers.py에서 계산된 실제 z 통계
                d[f"out_{split}_z_avg"] = m.get("z_avg")
                d[f"out_{split}_z_std"] = m.get("z_std")
                d[f"out_{split}_z_min"] = m.get("z_min")
                d[f"out_{split}_z_max"] = m.get("z_max")
        return d

    rows = []
    def _row(tag, score):
        d = dict(
            setting=tag,
            test_acc=float(score["test_acc"]),
            test_f1=float(score.get("test_f1", float("nan"))),
        )
        # outlier 메타(있다면) 일부 표시
        if "outlier_meta" in score and isinstance(score["outlier_meta"], dict):
            d.update(_flatten_outlier_meta(score["outlier_meta"]))
        return d

    rows.append(_row("CLEAN",   score_c))
    rows.append(_row("OUTLIER", score_o))
    df_results = pd.DataFrame(rows)

    return ([hist_c, hist_o], ["CLEAN", "OUTLIER"], df_results)
