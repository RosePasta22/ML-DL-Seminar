# robustloss/__init__.py
# Bind names explicitly into the package namespace

from . import loss_functions as _lf
from . import train_many    as _tm
from . import schemas       as _sch
from . import models        as _mdl

# 선택: 없을 수도 있으니 안전하게
try:
    from . import noise_types as _nt
except Exception:
    _nt = None

# === Loss functions ===
make_loss  = _lf.make_loss
ce_loss    = _lf.ce_loss
gce_loss   = _lf.gce_loss
focal_loss = _lf.focal_loss
cce_loss   = _lf.cce_loss
scce_loss  = _lf.scce_loss

# === Training ===
run_experiment     = _tm.run_experiment
run_clean_vs_noise = _tm.run_clean_vs_noise
plot_history       = _tm.plot_history
suggest_hparams    = getattr(_tm, "suggest_hparams", None)

# === Schemas (클래스 바인딩) ===
DatasetSchema = _sch.DatasetSchema
TaskType      = _sch.TaskType

# === Models / Noise ===
build_model = _mdl.build_model
NoiseConfig = (_nt.NoiseConfig if _nt else None)

# === Clean / Noise Compare ===
pct_drop = _tm.pct_drop

__all__ = [
    "make_loss","ce_loss","gce_loss","focal_loss","cce_loss","scce_loss",
    "run_experiment","run_clean_vs_noise","plot_history","suggest_hparams",
    "DatasetSchema","TaskType","build_model","NoiseConfig",
]
