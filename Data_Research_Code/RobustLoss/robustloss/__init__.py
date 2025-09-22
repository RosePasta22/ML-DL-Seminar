# robustloss/__init__.py
# Bind names explicitly into the package namespace

from . import loss_functions as _lf
from . import train_many    as _tm
from . import schemas       as _sch
from . import models        as _mdl

# === Noise ===
try:
    from . import noise_types as _nt
except Exception:
    _nt = None

# === Outliers (신규) ===
try:
    from . import outliers as _ol
    from . import apply_outliers as _ao
except Exception:
    _ol = None
    _ao = None

# === Loss functions ===
make_loss  = _lf.make_loss
ce_loss    = _lf.ce_loss
gce_loss   = _lf.gce_loss
focal_loss = _lf.focal_loss
cce_loss   = _lf.cce_loss
scce_loss  = _lf.scce_loss

# === Training ===
run_experiment        = _tm.run_experiment
run_clean_vs_noise    = _tm.run_clean_vs_noise
run_clean_vs_outlier  = _tm.run_clean_vs_outlier

plot_history    = _tm.plot_history
suggest_hparams = getattr(_tm, "suggest_hparams", None)

# === Schemas (클래스 바인딩) ===
DatasetSchema = _sch.DatasetSchema
TaskType      = _sch.TaskType

# === Models ===
build_model = _mdl.build_model

# === Noise ===
NoiseConfig = (_nt.NoiseConfig if _nt else None)

# === Outliers 공개 심볼 ===
OutlierConfig      = (_ol.OutlierConfig if _ol else None)
apply_outliers     = (_ao.apply_outliers if _ao else None)

# === Clean / Modified_Data Compare ===
pct_drop = _tm.pct_drop

__all__ = [
    # loss
    "make_loss","ce_loss","gce_loss","focal_loss","cce_loss","scce_loss",
    # training
    "run_experiment","run_clean_vs_noise","run_clean_vs_outlier","plot_history","suggest_hparams",
    # schema / model
    "DatasetSchema","TaskType","build_model","NoiseConfig", "OutlierConfig"
    # utils
    "pct_drop",
    # outliers
    "apply_outliers","xy_to_df","df_to_xy","summarize_outliers",
]
