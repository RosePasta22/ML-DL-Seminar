# robustloss/__init__.py
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# --- Version ---
try:
    __version__ = _pkg_version("robustloss-lab")  # 배포 이름 기준
except PackageNotFoundError:
    __version__ = "0.0.0"

# --- Required Submodules ---
from . import loss_functions as _lf
from . import train_many as _tm
from . import schemas as _sch
from . import models as _mdl

# --- Optional Submodules ---
try:
    from . import noise_types as _nt
except ImportError:
    _nt = None

try:
    from . import outliers as _ol
    from . import apply_outliers as _ao
except ImportError:
    _ol = None
    _ao = None

# === Public API Binding ===
# loss
make_loss  = _lf.make_loss
ce_loss    = _lf.ce_loss
gce_loss   = _lf.gce_loss
focal_loss = _lf.focal_loss
cce_loss   = _lf.cce_loss
scce_loss  = _lf.scce_loss

# training
run_experiment       = _tm.run_experiment
run_clean_vs_noise   = _tm.run_clean_vs_noise
run_clean_vs_outlier = _tm.run_clean_vs_outlier
plot_history         = _tm.plot_history
suggest_hparams      = getattr(_tm, "suggest_hparams", None)

# schemas / models
DatasetSchema = _sch.DatasetSchema
TaskType      = _sch.TaskType
build_model   = _mdl.build_model

# optional: noise / outliers
NoiseConfig      = (_nt.NoiseConfig if _nt else None)
OutlierConfig    = (_ol.OutlierConfig if _ol else None)
apply_outliers   = (_ao.apply_outliers if _ao else None)

# utils
pct_drop = _tm.pct_drop

__all__ = [
    "__version__",
    # loss
    "make_loss","ce_loss","gce_loss","focal_loss","cce_loss","scce_loss",
    # training
    "run_experiment","run_clean_vs_noise","run_clean_vs_outlier","plot_history","suggest_hparams",
    # schema / model
    "DatasetSchema","TaskType","build_model",
    # optional
    "NoiseConfig","OutlierConfig","apply_outliers",
    # utils
    "pct_drop",
]
