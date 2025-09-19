# loss_functions.py
import torch
import torch.nn.functional as F
from typing import Callable

__all__ = ["ce_loss", "gce_loss", "focal_loss", "cce_loss", "scce_loss", "make_loss"]

def _log_pt(logits: torch.Tensor, targets: torch.Tensor):
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
    pt = log_pt.exp()
    return log_pt, pt

def ce_loss(logits, targets, reduction="mean", class_weight=None):
    w = None
    if class_weight is not None:
        w = torch.as_tensor(class_weight, device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, targets, weight=w, reduction=reduction)

def focal_loss(logits, targets, gamma: float = 2.0, alpha=None, reduction="mean", eps=1e-8):
    log_pt, pt = _log_pt(logits, targets)
    modulating = (1 - pt).clamp(min=eps) ** gamma
    if alpha is None:
        a = 1.0
    else:
        if isinstance(alpha, (list, tuple)):
            a = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)[targets]
        elif isinstance(alpha, torch.Tensor):
            a = alpha.to(logits.device, logits.dtype)[targets]
        else:
            a = torch.full_like(pt, float(alpha))
    loss = -a * modulating * log_pt
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss

def gce_loss(logits, targets, q: float = 0.7, reduction="mean", eps=1e-12):
    _, pt = _log_pt(logits, targets)
    pt = pt.clamp(min=eps)
    loss = -pt.log() if q == 0.0 else (1.0 - pt.pow(q)) / q
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss

def cce_loss(logits, targets, eps: float = 1e-3, reduction="mean"):
    # 소프트맥스 확률
    probs = torch.softmax(logits, dim=1)
    # 정답 클래스 확률 p_t
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # epsilon 아래로 떨어지지 않도록 clamp
    pt = torch.clamp(pt, min=eps)
    loss = -torch.log(pt)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def scce_loss(logits, targets, eps: float = 1e-3, reduction: str = "mean"):
    # 소프트맥스 확률
    probs = F.softmax(logits, dim=1)                          # (N, C)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)     # (N,)

    eps_t = torch.as_tensor(eps, device=logits.device, dtype=logits.dtype)

    # 원식: -log(pt + eps)
    # 음수 손실( pt≈1 일 때 -log(1+eps) < 0 ) 방지를 위해 상한을 1-1e-8로 제한
    pt_eps = torch.clamp(pt + eps_t, max=1.0 - 1e-8)

    loss = -torch.log(pt_eps)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def make_loss(name: str, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    n = name.lower()
    if n in ("ce", "crossentropy", "cross_entropy"): return lambda l, t: ce_loss(l, t, **kwargs)
    if n == "gce":   return lambda l, t: gce_loss(l, t, **kwargs)
    if n in ("focal", "focal_loss"): return lambda l, t: focal_loss(l, t, **kwargs)
    if n in ("cce", "capped_ce"):    return lambda l, t: cce_loss(l, t, **kwargs)
    if n in ("scce", "softcce", "soft_capped_ce"): return lambda l, t: scce_loss(l, t, **kwargs)
    raise ValueError(f"Unknown loss: {name}")
