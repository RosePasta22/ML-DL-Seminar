# robustloss/__init__.py
# Bind names explicitly into the package namespace (fixes import issues)

from . import loss_functions as _lf
from . import train_many as _tm
from .schemas import DatasetSchema, TaskType
from .models import build_model
from .noise_types import NoiseConfig

# === Loss functions ===
make_loss   = _lf.make_loss
ce_loss     = _lf.ce_loss
gce_loss    = _lf.gce_loss
focal_loss  = _lf.focal_loss
cce_loss    = _lf.cce_loss
scce_loss   = _lf.scce_loss

# === Training ===
run_experiment     = _tm.run_experiment
run_clean_vs_noise = _tm.run_clean_vs_noise
plot_history       = _tm.plot_history
suggest_hparams    = _tm.suggest_hparams

__all__ = [
    "make_loss", "ce_loss", "gce_loss", "focal_loss", "cce_loss", "scce_loss",
    "run_experiment", "run_clean_vs_noise", "plot_history", "suggest_hparams",
    "DatasetSchema", "TaskType", "build_model", "NoiseConfig",
]
