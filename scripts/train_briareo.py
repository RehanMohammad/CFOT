#!/usr/bin/env python3
# ST-GCN / MSG3D training on Briareo with optional CFOT + robust viz

from __future__ import annotations

import os
import math
import random
import inspect
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.viz import AdjLogger, plot_per_delta_heatmaps, plot_temporal_heatmap
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, List, Tuple, Optional, Sequence, Union
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


# Project modules
from dataset.Briareo import BriareoDataset, build_briareo_label_map, briareo_collate
from matplotlib.ticker import FixedLocator, FixedFormatter   
# sklearn (optional)
try:
    from sklearn.metrics import classification_report
except Exception:
    classification_report = None

# ------------------------------
# Minimal, robust viz helper (internal)
# ------------------------------
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt

def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)

def _save_png(mat, save_path, title, vmax=None, xlabel="", ylabel=""):
    A = _np(mat).astype(float)
    A = np.clip(A, 0.0, None)                    # drop negatives only
    if vmax is None:
        vmax = float(min(1.0, np.nanmax(A) + 1e-8))  # adaptive top, but ≤1
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(A, vmin=0.0, vmax=vmax, interpolation="nearest")
    # im = ax.imshow(A, norm=colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0), interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    V = A.shape[-1]
    ticks = np.arange(V)
    labels = [str(i+1) for i in ticks]
    ax.set_xlim(-0.5, V-0.5); ax.set_ylim(V-0.5, -0.5); ax.set_aspect("equal")
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter(labels))
    ax.set_xticks(np.arange(-0.5, V, 1), minor=True); ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for sp in ax.spines.values(): sp.set_visible(False)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


class AdjLogger:
    """
    Keeps per-epoch accumulators for spatial (V,V) and temporal ((D,V,V)) adjacencies.
    Accepts tensors or numpy; accepts (V,V) or (D,V,V) for temporal and broadcasts.
    API compatible with existing calls: begin_epoch(), update(...), end_epoch(...),
    snapshot_epoch0(...).
    """
    def __init__(self, V: int, deltas: Optional[Sequence[int]] = None, save_every: int = 10):
        self.V = int(V)
        self.deltas = list(deltas) if deltas else [1]
        self.D = len(self.deltas)
        self.save_every = int(save_every)
        self._zeros_VV = np.zeros((self.V, self.V), dtype=np.float32)
        self._zeros_DVV = np.zeros((self.D, self.V, self.V), dtype=np.float32)
        self.begin_epoch()  # init accumulators

    def begin_epoch(self):
        self.spatial_sum = np.zeros((self.V, self.V), dtype=np.float32)
        self.spatial_count = 0
        self.temporal_sum = np.zeros((self.D, self.V, self.V), dtype=np.float32)
        self.temporal_count = 0
        self._last_spatial = self._zeros_VV.copy()
        self._last_temporal = self._zeros_DVV.copy()

    def _ensure_VV(self, x):
        if x is None:
            return self._zeros_VV.copy()
        A = _np(x)
        if A.ndim == 2 and A.shape == (self.V, self.V):
            return A
        if A.ndim >= 3 and A.shape[-2:] == (self.V, self.V):
            return A.mean(axis=tuple(range(A.ndim - 2)))
        raise ValueError(f"spatial matrix must end with (V,V); got {A.shape}")

    def _ensure_DVV(self, x):
        if x is None:
            return self._zeros_DVV.copy()
        A = _np(x)
        if A.ndim == 2 and A.shape == (self.V, self.V):
            return np.broadcast_to(A, (self.D, self.V, self.V)).copy()
        if A.ndim == 3 and A.shape[-2:] == (self.V, self.V):
            if A.shape[0] == self.D:
                return A
            m = A.mean(axis=0)
            return np.broadcast_to(m, (self.D, self.V, self.V)).copy()
        if A.ndim >= 4 and A.shape[-2:] == (self.V, self.V):
            m = A.mean(axis=tuple(range(A.ndim - 2)))
            return np.broadcast_to(m, (self.D, self.V, self.V)).copy()
        raise ValueError(f"temporal matrix must end with (V,V) or (D,V,V); got {A.shape}")

    def update(self, spatial_list: Optional[List[Any]] = None,
               temporal: Optional[Any] = None,
               temporal_by_delta: Optional[Any] = None):
        try:
            if spatial_list:
                mats = [_np(m) for m in spatial_list if m is not None]
                if mats:
                    VV = np.mean(np.stack([self._ensure_VV(m) for m in mats], axis=0), axis=0)
                    self.spatial_sum += VV; self.spatial_count += 1; self._last_spatial = VV
            if temporal_by_delta is not None:
                DVV = self._ensure_DVV(temporal_by_delta)
            elif temporal is not None:
                DVV = self._ensure_DVV(temporal)
            else:
                DVV = self._zeros_DVV.copy()
            self.temporal_sum += DVV; self.temporal_count += 1; self._last_temporal = DVV
        except Exception as e:
            print(f"[viz] update skipped (coerced): {e}")

    def _means(self):
        if self.spatial_count > 0:
            sp = self.spatial_sum / float(self.spatial_count)
        else:
            sp = self._last_spatial
        if self.temporal_count > 0:
            tm = self.temporal_sum / float(self.temporal_count)
        else:
            tm = self._last_temporal
        return sp, tm

    def snapshot_epoch0(self, spatial=None, temporal=None, temporal_by_delta=None, out_dir: Path = Path(".")):
        out_dir = Path(out_dir)
        try:
            sp = self._ensure_VV(spatial) if spatial is not None else self._zeros_VV
            tm = self._ensure_DVV(temporal_by_delta if temporal_by_delta is not None else temporal)
            np.save(out_dir / "spatial_epoch0.npy", sp)
            np.save(out_dir / "temporal_by_delta_epoch0.npy", tm)

            _save_png(
                sp,
                out_dir / "spatial_epoch_000.png",
                "spatial — epoch 000",
                vmax=None,
                xlabel="target",
                ylabel="source",
            )
            for d_idx, d in enumerate(self.deltas):
                _save_png(
                    tm[d_idx],
                    out_dir / f"temporal_epoch_000_delta_{d}.png",
                    f"temporal — epoch 000 — Δ={d}",
                    vmax=None,
                    xlabel="target joints (t+Δ)",
                    ylabel="source joints (t)",
                )
            print("[viz] Saved epoch0 snapshots (spatial/temporal).")
        except Exception as e:
            print(f"[viz] epoch0 snapshot save failed: {e}")


    def end_epoch(self, epoch: int, out_dir: Path = Path("."), vmax: float = 0.1):
        out_dir = Path(out_dir)
        try:
            sp, tm = self._means()
            # np.save(out_dir / f"spatial_adj_epoch{epoch}.npy", sp)
            np.save(out_dir / f"temporal_by_delta_epoch{epoch}.npy", tm)
            _save_png(sp, out_dir / f"spatial_adj_epoch_{epoch:03d}.png",
                      f"spatial — epoch {epoch:03d}", vmax, "target joints", "source joints")
            for d_idx, d in enumerate(self.deltas):
                _save_png(tm[d_idx], out_dir / f"temporal_adj_epoch_{epoch:03d}_delta_{d}.png",
                          f"temporal — epoch {epoch:03d} — Δ={d}", vmax,
                          "target joints (t+Δ)", "source joints (t)")
        except Exception as e:
            print(f"[viz] end_epoch failed: {e}")

        

viz_logger = None  # global used inside epoch fns

# ------------------------------
# Utilities
# ------------------------------

def _extract_temporal_from_model(model, V: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (avg_TVV, by_delta_DVV) as numpy or (None,None).
    - avg_TVV: [T,V,V] or [V,V] representing per-frame or averaged temporal adj
    - by_delta_DVV: [D,V,V] for the configured deltas
    """
    def to_np(x):
        if x is None: return None
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
        return np.asarray(x)

    avg = to_np(getattr(model, "cfot_last_avg_adj", None))          # [T,V,V] or [V,V]
    byD = to_np(getattr(model, "cfot_last_adj_by_delta", None))     # [D,V,V] or [T-1,D,V,V] or similar

    # Reduce time if needed
    if avg is not None and avg.ndim >= 3 and avg.shape[-2:] == (V, V):
        # keep as [T,V,V] for logger to collapse; else collapse here to [V,V]
        pass
    elif avg is not None and avg.shape == (V, V):
        pass
    else:
        avg = None

    # Collapse any time dimension on by-delta to [D,V,V]
    if byD is not None and byD.shape[-2:] == (V, V):
        if byD.ndim == 4:
            # [T-1, D, V, V] -> mean over time
            byD = byD.mean(axis=0)
        elif byD.ndim == 3:
            # already [D, V, V]
            pass
        elif byD.ndim == 2 and byD.shape == (V, V):
            byD = byD[None, ...]   # D=1
        else:
            byD = None
    else:
        byD = None

    return avg, byD


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def override(cfg: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    for k, v in kwargs.items():
        if v is not None:
            cfg[k] = v
    return cfg

def accuracy_simple(logits: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = (pred == target).sum().item()
        total = target.numel()
        return 0.0 if total == 0 else correct / total

def accuracy_at_k(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> Dict[int, float]:
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = {}
    for k in ks:
        k = min(k, logits.size(1))
        res[k] = correct[:k].reshape(-1).float().sum(0).item() * 100.0 / target.size(0)
    return res

def safe_confmat_and_report(y_true: List[int], y_pred: List[int], n_classes: int) -> Tuple[np.ndarray, str]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    if classification_report is not None:
        try:
            report = classification_report(y_true, y_pred, digits=4)
        except Exception:
            report = ""
    else:
        report = ""
    return cm, report

def write_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def save_numpy_csv(path: Path, array: np.ndarray, header: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), array, delimiter=",", fmt="%d", header=header)

# ------------------------------
# Split helpers
# ------------------------------

def _read_ann_lines(ann_path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"Bad line in {ann_path}: '{s}'")
            path_part = " ".join(parts[:-1])
            label = int(parts[-1])
            items.append((path_part, label))
    if not items:
        raise ValueError(f"No items found in {ann_path}")
    return items

def stratified_split(items, val_ratio: float, seed: int):
    by_label: Dict[int, List[str]] = {}
    for p, y in items:
        by_label.setdefault(y, []).append(p)
    rng = random.Random(seed)
    train_items, val_items = [], []
    for y, paths in by_label.items():
        rng.shuffle(paths)
        n = len(paths)
        if n == 1:
            nv = 0
        else:
            nv = int(round(n * val_ratio))
            nv = max(1, min(n - 1, nv))
        val_paths = paths[:nv]
        tr_paths  = paths[nv:]
        val_items.extend([(p, y) for p in val_paths])
        train_items.extend([(p, y) for p in tr_paths])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items

def write_ann_file(path: Path, items: List[Tuple[str,int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, y in items:
            f.write(f"{p} {y}\n")

# ------------------------------
# Viz shape helpers
# ------------------------------

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)

def _coerce_to_VV(x, V: int):
    A = _to_numpy(x)
    if A is None: return None
    if A.ndim == 2 and A.shape == (V, V): return A
    if A.ndim == 3 and A.shape[-2:] == (V, V): return A.mean(axis=0)
    if A.ndim == 4 and A.shape[-2:] == (V, V): return A.mean(axis=(0, 1))
    return None

def _coerce_to_DVV(x, V: int):
    B = _to_numpy(x)
    if B is None: return None
    if B.ndim == 3 and B.shape[-2:] == (V, V): return B
    if B.ndim == 4 and B.shape[-2:] == (V, V): return B.mean(axis=0)
    if B.ndim == 5 and B.shape[-2:] == (V, V): return B.mean(axis=(0, 1))
    return None

def _ensure_DVV_any(x, D: int, V: int):
    A = _to_numpy(x)
    if A is None:
        return np.zeros((D, V, V), dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    if A.ndim == 2 and A.shape == (V, V):
        return np.repeat(A[None, ...], D, axis=0)
    if A.ndim == 3 and A.shape[-2:] == (V, V):
        if A.shape[0] == D:
            return A
        Amean = A.mean(axis=0)
        return np.repeat(Amean[None, ...], D, axis=0)
    if A.ndim >= 4 and A.shape[-2:] == (V, V):
        Amean = A.mean(axis=tuple(range(A.ndim - 2)))
        return np.repeat(Amean[None, ...], D, axis=0)
    return np.zeros((D, V, V), dtype=np.float32)

# ------------------------------
# CFOT diagnostics
# ------------------------------

def cfot_diagnostic(model: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    info = {"has_cfot": 0.0, "strength": float(getattr(model, "cfot_strength_frac", 0.0)),
            "beta": float(getattr(model, "cfot_beta", 1.0)), "effect_mae": 0.0, "applied": 0.0}
    if not hasattr(model, "cfot_module") or getattr(model, "cfot_module", None) is None:
        return info
    info["has_cfot"] = 1.0
    if info["strength"] <= 0.0 or info["beta"] == 0.0:
        return info
    was_training = model.training
    orig_strength = float(getattr(model, "cfot_strength_frac", 0.0))
    try:
        model.eval()
        with torch.no_grad():
            logits_on = model(x)
            setattr(model, "cfot_strength_frac", 0.0)
            logits_off = model(x)
        mae = (logits_on - logits_off).abs().mean().item()
        info["effect_mae"] = float(mae)
        info["applied"] = 1.0 if mae > 1e-8 else 0.0
    finally:
        setattr(model, "cfot_strength_frac", orig_strength)
        model.train(was_training)
    return info

def _ensure_DVV(x, D: int, V: int):
    """Coerce many shapes to (D,V,V) for viz logger."""
    A = _to_numpy(x)
    if A is None:
        return np.zeros((D, V, V), dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)

    if A.ndim == 2 and A.shape == (V, V):
        return np.repeat(A[None, ...], D, axis=0)

    if A.ndim == 3 and A.shape[-2:] == (V, V):
        if A.shape[0] == D:
            return A
        Amean = A.mean(axis=0)
        return np.repeat(Amean[None, ...], D, axis=0)

    if A.ndim >= 4 and A.shape[-2:] == (V, V):
        Amean = A.mean(axis=tuple(range(A.ndim - 2)))
        return np.repeat(Amean[None, ...], D, axis=0)

    return np.zeros((D, V, V), dtype=np.float32)


# ------------------------------
# Train / Val epochs
# ------------------------------

def run_train_epoch(model: nn.Module, loader: DataLoader, optimizer, device, criterion,
                    epoch: Optional[int] = None, writer: Optional[SummaryWriter] = None,
                    do_cfot_check: bool = False, V: int = 21) -> Dict[str, float]:
    model.train(True)
    tot_loss = 0.0
    tot_acc  = 0.0
    tot_n    = 0
    first_batch_done = False

    for batch in loader:
        x = batch["data"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if do_cfot_check and (not first_batch_done) and (epoch is not None):
            diag = cfot_diagnostic(model, x)
            if writer is not None:
                writer.add_scalar("cfot/enabled", diag["has_cfot"], epoch)
                writer.add_scalar("cfot/strength_frac", diag["strength"], epoch)
                writer.add_scalar("cfot/effect_mae", diag["effect_mae"], epoch)
            if diag["has_cfot"] < 0.5:
                print(f"[cfot] epoch {epoch}: NO cfot_module present.")
            else:
                status = "ACTIVE" if diag["applied"] > 0.5 else "INACTIVE"
                print(f"[cfot] epoch {epoch}: strength={diag['strength']:.3f}, "
                      f"beta={diag['beta']:.2f}, effect_mae={diag['effect_mae']:.4e} -> {status}")
            first_batch_done = True

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        def _num(x): 
            import numpy as np, torch
            if x is None: return None
            if isinstance(x, torch.Tensor): x = x.detach().float().cpu().numpy()
            return (x.shape, float(x.min()), float(x.max()), float(x.mean()))
        # print("[chk] avg_adj", _num(getattr(model,"cfot_last_avg_adj", None)))
        # print("[chk] by_delta", _num(getattr(model,"cfot_last_adj_by_delta", None)))
        cls_loss = criterion(logits, y)

        reg = 0.0
        if hasattr(model, "cfot_module") and hasattr(model.cfot_module, "regularization_loss"):
            reg = model.cfot_module.regularization_loss()
        loss = cls_loss + reg

        # viz accumulation (safe, no unbound names)
        if viz_logger is not None:
            try:
                D = len(getattr(viz_logger, "deltas", []) or [])
                D = max(D, 1)
                # Spatial (ST-GCN edge_importance), summed over stages
                sp_list = []
                if hasattr(model, "edge_importance"):
                    sp_list = [w.detach().float().cpu().numpy() for w in model.edge_importance]

                # Temporal from CFOT buffers
                avg_TVV, by_delta = _extract_temporal_from_model(model, V)
                # Coerce shapes to what the logger expects
                temporal = None
                if avg_TVV is not None:
                    if avg_TVV.ndim == 3 and avg_TVV.shape[-2:] == (V, V):
                        temporal = avg_TVV.mean(axis=0)                      # [V,V]
                    elif avg_TVV.ndim == 2 and avg_TVV.shape == (V, V):
                        temporal = avg_TVV                                     # [V,V]

                temporal_by_delta = _ensure_DVV(by_delta, D=D, V=V) if by_delta is not None else None

                # Tile single temporal [V,V] to [D,V,V] so grids render side-by-side
                temporal_DVV = None
                if temporal is not None:
                    temporal_DVV = np.repeat(temporal[None, ...], D, axis=0)

                viz_logger.update(
                    spatial_list=sp_list,
                    temporal=temporal_DVV,            # optional
                    temporal_by_delta=temporal_by_delta  # preferred if present
                )
            except Exception as e:
                print(f"[viz] update skipped: {e}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = y.size(0)
        tot_loss += float(loss.item()) * bs
        tot_acc  += accuracy_simple(logits, y) * bs
        tot_n    += bs

    denom = max(1, tot_n)
    return {"loss": tot_loss/denom, "acc": tot_acc/denom}

def run_val_epoch(model: nn.Module, loader: DataLoader, device, criterion,
                  num_classes: Optional[int],
                  epoch: Optional[int] = None, writer: Optional[SummaryWriter] = None) -> Dict[str, Any]:
    model.train(False)
    tot_loss = tot_acc = 0.0
    tot_n = 0
    top1_meter = top5_meter = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["data"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x)
            def _num(x): 
                if x is None: return None
                if isinstance(x, torch.Tensor): x = x.detach().float().cpu().numpy()
                return (x.shape, float(x.min()), float(x.max()), float(x.mean()))
            # print("[chk] avg_adj", _num(getattr(model,"cfot_last_avg_adj", None)))
            # print("[chk] by_delta", _num(getattr(model,"cfot_last_adj_by_delta", None)))
            loss = criterion(logits, y)
            bs = y.size(0)
            tot_loss += float(loss.item()) * bs
            tot_acc  += accuracy_simple(logits, y) * bs
            tot_n    += bs
            accs = accuracy_at_k(logits, y, ks=(1, 5))
            top1_meter += accs[1] * bs / 100.0
            top5_meter += accs[5] * bs / 100.0
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    denom = max(1, tot_n)
    loss_epoch = tot_loss / denom
    acc_epoch  = tot_acc / denom
    top1 = 100.0 * (top1_meter / denom)
    top5 = 100.0 * (top5_meter / denom)

    nc = int(num_classes) if num_classes is not None else int(max(y_true) + 1)
    cm, cls_report_txt = safe_confmat_and_report(y_true, y_pred, n_classes=nc)

    per_class_recall = []
    for c in range(nc):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        recall_c = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        per_class_recall.append(recall_c)

    if writer is not None and epoch is not None:
        writer.add_scalar("topk/val_top1", top1, epoch)
        writer.add_scalar("topk/val_top5", top5, epoch)

    return {"loss": loss_epoch, "acc": acc_epoch, "top1": top1, "top5": top5,
            "cm": cm, "per_class_recall": per_class_recall, "report": cls_report_txt}

# ------------------------------
# CFOT schedule
# ------------------------------

def make_cfot_frac_fn(start: int, warm: int):
    def f(epoch_1based: int) -> float:
        if epoch_1based < start:
            return 0.0
        if warm <= 0:
            return 1.0
        return min(1.0, (epoch_1based - start + 1) / float(warm))
    return f

# ------------------------------
# Model builders
# ------------------------------

def _safe_build(cls, base_kwargs: Dict[str, Any], maybe_kwargs: Dict[str, Any]):
    try:
        return cls(**base_kwargs, **maybe_kwargs)
    except TypeError:
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys())
        cleaned = {k: v for k, v in maybe_kwargs.items() if k in allowed}
        try:
            return cls(**base_kwargs, **cleaned)
        except TypeError:
            return cls(**base_kwargs)

def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model_name = str(cfg.get("model", "stgcn")).lower()
    feat = cfg.get("feat", "xyz+vel")
    in_ch = 3 if feat == "xyz" else 6
    V = 21

    if model_name in ("msg3d_2s","twostreammsg3d","msg3d_two_stream","2s_msg3d"):
        from models.MSG3D.msg3d_two_stream import TwoStreamMSG3D
        model = TwoStreamMSG3D(
            num_class=int(cfg.get("num_class",14)),
            num_point=22, num_person=1,
            graph="utils.graph.Graph",
            graph_args={"layout":"briareo",
                        "strategy":cfg.get("graph_strategy","spatial"),
                        "max_hop": int(cfg.get("max_hop",2))},
            in_channels=in_ch,
            fusion=cfg.get("fusion","mean_logits"),
            enable_cfot=bool(cfg.get("enable_cfot", False)),
            cfot_hidden=int(cfg.get("cfot_hidden", 64)),
            cfot_topk=int(cfg.get("cfot_topk", 3)),
            cfot_tau=float(cfg.get("cfot_tau", 0.7)),
            cfot_iters=int(cfg.get("cfot_iters", 10)),
            cfot_beta=float(cfg.get("cfot_beta", 1.0)),
            cfot_deltas=cfg.get("cfot_deltas", [1,2]),
            dropout=float(cfg.get("dropout",0.1)),
        ).to(device)
        return model

    if model_name == "agcn":
        from models.agcn.agcn import Model as AGCN
        return AGCN(
            in_channels=in_ch,
            num_class=int(cfg.get("num_class", 14)),
            graph_args={"layout":"briareo",
                        "strategy":cfg.get("graph_strategy","spatial"),
                        "max_hop": int(cfg.get("max_hop",1))},
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.15)),
        ).to(device)

    if model_name == "ctrgcn":
        from models.ctrgcn.ctrgcn import Model as CTRGCN
        return CTRGCN(
            in_channels=in_ch,
            num_class=int(cfg.get("num_class", 14)),
            graph_args={"layout":"briareo",
                        "strategy":cfg.get("graph_strategy","spatial"),
                        "max_hop": int(cfg.get("max_hop",1))},
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.10)),
        ).to(device)

    # default: ST-GCN
    from models.stgcn.stgcn import Model as STGCN

    base_kwargs = dict(
        in_channels=in_ch,
        num_class=int(cfg.get("num_class", 14)),
        graph_args={"layout":"briareo",
                    "strategy":cfg.get("graph_strategy","spatial"),
                    "max_hop": int(cfg.get("max_hop",1))},
        edge_importance_weighting=True,
        dropout=float(cfg.get("dropout", 0.05)),
    )

    maybe_kwargs = dict(
        enable_cfot=bool(cfg.get("enable_cfot", False)),
        cfot_type=cfg.get("cfot_type", "adaptive"),
        cfot_deltas=cfg.get("cfot_deltas", [1, 2]),
        cfot_hidden=int(cfg.get("cfot_hidden", 64)),
        cfot_iters=int(cfg.get("cfot_iters", 10)),
        cfot_tau=float(cfg.get("cfot_tau", 0.7)),
        cfot_topk=int(cfg.get("cfot_topk", 3)),
        cfot_beta=float(cfg.get("cfot_beta", 1.0)),
        cfot_inject=cfg.get("cfot_inject", "after1"),
        temporal_kernel_size=int(cfg.get("temporal_kernel_size", 9)),
        cfot_sparsify=cfg.get("cfot_sparsify", "topk"),
        cfot_keep_mass=float(cfg.get("cfot_keep_mass", 0.90)),
        cfot_min_k=int(cfg.get("cfot_min_k", 2)),
        cfot_max_k=int(cfg.get("cfot_max_k", None) or V),
        cfot_affinity=cfg.get("cfot_affinity","euclid"),
        cfot_euclid_scale=float(cfg.get("cfot_euclid_scale",1.0)),
        cfot_cosine_eps=float(cfg.get("cfot_cosine_eps",1e-6)),
        cfot_pos_weight=float(cfg.get("cfot_pos_weight",1.0)),
        cfot_vel_weight=float(cfg.get("cfot_vel_weight",0.2)),
    )

    model = _safe_build(STGCN, base_kwargs, maybe_kwargs).to(device)
    return model

# ------------------------------
# Per-class Δ=1 random-frame heatmaps (simple exporter)
# ------------------------------

def _pick_one_per_class(dataset, num_classes: int) -> Dict[int, int]:
    picked = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        y = int(item["label"] if isinstance(item, dict) else item[1])
        if y not in picked:
            picked[y] = idx
            if len(picked) == num_classes:
                break
    return picked

@torch.no_grad()
def export_delta1_random_frame_heatmaps(dataset, model, device, out_dir, epoch,
                                        max_per_class=1, track_class=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    rng = random.Random(0 + epoch)

    per_cls_idx = {}
    for idx in range(len(dataset)):
        y = int(dataset.get_label(idx)) if hasattr(dataset, "get_label") else int(dataset[idx]["label"])
        if (track_class is not None) and (y != track_class):
            continue
        per_cls_idx.setdefault(y, []).append(idx)

    classes = sorted(per_cls_idx.keys())
    if track_class is not None and not classes:
        print(f"[viz] track_class={track_class} not found in dataset; skipping.")
        return

    for y in classes:
        cls_idxs = per_cls_idx[y]
        rng.shuffle(cls_idxs)
        take = cls_idxs[:max_per_class]
        cls_dir = out_dir / f"class_{y}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        for idx in take:
            sample = dataset[idx]
            x = torch.as_tensor(sample["data"]).unsqueeze(0).to(device)  # [1,C,T,V]
            _ = model(x)  # populate model.cfot_last_adj_by_delta if present

            T = x.shape[2]
            t = rng.randint(0, max(0, T - 2))
            H = None
            A = getattr(model, "cfot_last_adj_by_delta", None)
            if A is not None:
                A = _np(A)
                if A.ndim == 3 and A.shape[0] in (T - 1, 1):
                    H = A[min(t, A.shape[0]-1)]
                elif A.ndim == 4:
                    if A.shape[0] == (T - 1):
                        H = A[t, 0]
                    elif A.shape[1] == (T - 1):
                        H = A[0, t]
            if H is None:
                H = np.zeros((21, 21), dtype=np.float32)

            V = H.shape[0]
            H = np.asarray(H, dtype=np.float32)
            # force [0, 1] scale
            H = np.clip(H, 0.0, 1.0)

            fig = plt.figure(figsize=(4, 4), dpi=200)
            ax = fig.add_subplot(111)
            im = ax.imshow(H, vmin=0.0, vmax=1.0, aspect="equal")

            # integer ticks 1..V (no 0.0/2.5/…)
            ticks = np.arange(V)
            ax.set_xticks(ticks); ax.set_xticklabels([str(i+1) for i in ticks])
            ax.set_yticks(ticks); ax.set_yticklabels([str(i+1) for i in ticks])
            ax.set_xlim(-0.5, V-0.5); ax.set_ylim(V-0.5, -0.5)

            ax.set_title(f"class {y} — epoch {epoch:03d} — t→t+1 (t={t})")
            ax.set_xlabel("target joints (t+1)")
            ax.set_ylabel("source joints (t)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def _plot_heatmap_01(A: np.ndarray, save_path: Path, title: str = ""):
    A = np.clip(np.asarray(A, dtype=float), 0.0, 1.0)
    V = A.shape[-1]
    fig, ax = plt.subplots(figsize=(4,4), dpi=200)
    im = ax.imshow(A, vmin=0.0, vmax=1.0, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title: ax.set_title(title)
    ticks = np.arange(V)
    labels = [str(i+1) for i in ticks]
    ax.set_xlim(-0.5, V-0.5); ax.set_ylim(V-0.5, -0.5); ax.set_aspect("equal")
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter(labels))
    ax.set_xticks(np.arange(-0.5, V, 1), minor=True); ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("target joint"); ax.set_ylabel("source joint")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)

@torch.no_grad()
def export_progression_pairs(dataset,
                             model: nn.Module,
                             device: torch.device,
                             out_root: Union[str, Path],
                             epoch: int,
                             pairs: Sequence[int] = (0, 9, 19, 29, 39, 49, 59, 69, 79),
                             delta_index: int = 0,  # Δ=1
                             one_sample_per_class: bool = True):
    """
    For each class, pick one random sample and draw heatmaps for (t, t+1) over `pairs`.
    Saves under: <out_root>/ep_{epoch:03d}/class_{c}/idx_<idx>_tXXX_tYYY.png
    Works for inputs shaped [N,C,T,V] or [N,C,T,V,M].
    """
    rng = random.Random(1000 + epoch)
    model.eval()
    out_root = Path(out_root) / f"ep_{epoch:03d}"
    out_root.mkdir(parents=True, exist_ok=True)

    # indices per class
    per_cls: Dict[int, List[int]] = {}
    for i in range(len(dataset)):
        y = int(dataset.get_label(i)) if hasattr(dataset, "get_label") else int(dataset[i]["label"])
        per_cls.setdefault(y, []).append(i)

    classes = sorted(per_cls.keys())
    for c in classes:
        idxs = per_cls[c]
        if not idxs:
            continue
        if one_sample_per_class:
            idxs = [rng.choice(idxs)]

        cls_dir = out_root / f"class_{c:02d}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        for idx in idxs:
            sample = dataset[idx]
            x = torch.as_tensor(sample["data"]).unsqueeze(0).to(device)  # add batch: [1, ...]
            dims = x.dim()
            if dims == 4:        # [1,C,T,V]
                _, _, T, V = x.shape
                has_m = False
            elif dims == 5:      # [1,C,T,V,M]
                _, _, T, V, _M = x.shape
                has_m = True
            else:
                # unexpected – skip
                continue

            for t in pairs:
                if t + 1 >= T:
                    continue

                # slice the two frames while preserving dims
                if has_m:
                    x_pair = x[:, :, t:t+2, :, :]      # [1,C,2,V,1]
                else:
                    x_pair = x[:, :, t:t+2, :]         # [1,C,2,V]

                # forward to populate CFOT buffers
                _ = model(x_pair)

                adj = getattr(model, "cfot_last_adj_by_delta", None)
                if isinstance(adj, torch.Tensor):
                    A = adj.detach().float().cpu().numpy()
                else:
                    A = None

                Vnow = V
                H = None
                if A is not None and A.ndim >= 3 and A.shape[-2:] == (Vnow, Vnow):
                    # common: [D,V,V]
                    if A.ndim == 3:
                        if delta_index < A.shape[0]:
                            H = A[delta_index]
                    elif A.ndim == 4:
                        # squeeze any singleton pre-V,V dims (e.g., [1,D,V,V] or [D,1,V,V])
                        squeeze_axes = tuple(i for i in range(A.ndim - 2) if A.shape[i] == 1)
                        A2 = np.squeeze(A, axis=squeeze_axes) if squeeze_axes else A
                        if A2.ndim == 3 and delta_index < A2.shape[0]:
                            H = A2[delta_index]
                        else:
                            # fallback: pick the first along the first axis
                            H = A2[0] if A2.ndim == 3 else None

                if H is None:
                    H = np.zeros((Vnow, Vnow), dtype=np.float32)

                # clamp to [0,1] and draw with 1..V ticks
                H = np.clip(np.asarray(H, dtype=np.float32), 0.0, 1.0)

                fig, ax = plt.subplots(figsize=(8, 8), dpi=400)
                im = ax.imshow(H, vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
                ticks = np.arange(Vnow)
                ax.set_xticks(ticks); ax.set_xticklabels([str(i+1) for i in ticks])
                ax.set_yticks(ticks); ax.set_yticklabels([str(i+1) for i in ticks])
                ax.set_xlim(-0.5, Vnow-0.5); ax.set_ylim(Vnow-0.5, -0.5)
                ax.set_xlabel("target joint (t+1)")
                ax.set_ylabel("source joint (t)")
                ax.set_title(f"class {c} — idx {idx} — t={t}→{t+1}")
                # grid between cells
                ax.set_xticks(np.arange(-0.5, Vnow, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, Vnow, 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=0.5)
                ax.tick_params(which="minor", bottom=False, left=False)
                for sp in ax.spines.values(): sp.set_visible(False)
                fig.colorbar(im, ax=ax)

                out_png = cls_dir / f"idx_{idx}_t{t:03d}_t{t+1:03d}.png"
                fig.tight_layout()
                fig.savefig(out_png)
                plt.close(fig)




# ------------------------------
# Main
# ------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)

    # Common overrides
    
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--split-seed", type=int, default=None, help="Seed for stratified split (default 42).")
    p.add_argument("--max-T", type=int, default=None)
    p.add_argument("--normalize", dest="normalize", action="store_true")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.set_defaults(normalize=None)

    p.add_argument("--feat", type=str, default=None, choices=["xyz","xyz+vel"])
    p.add_argument("--num-class", type=int, default=None)
    p.add_argument("--in-ch", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--graph-strategy", type=str, default=None)
    p.add_argument("--max-hop", type=int, default=None)

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--aug", type=str, default=None)  # 'true'/'false'
    p.add_argument("--temporal-mode", type=str, default=None, choices=["crop_repeat","interp"])
    p.add_argument("--model", type=str, default=None, choices=["stgcn","msg3d_2s","twostreammsg3d","msg3d_two_stream","2s_msg3d","agcn","ctrgcn"])

    # CFOT scheduling
    p.add_argument("--cfot-start-epoch", type=int, default=None)
    p.add_argument("--cfot-warmup-epochs", type=int, default=None)

    # CFOT toggles
    p.add_argument("--enable-cfot",  dest="enable_cfot", action="store_true")
    p.add_argument("--disable-cfot", dest="enable_cfot", action="store_false")
    p.set_defaults(enable_cfot=None)
    p.add_argument("--strict-cfot", action="store_true")

    # Eval-only shortcut
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--load', type=str, default=None)
    p.add_argument('--ann-test', type=str, default=None)

    # Viz export cadence
    p.add_argument("--export-adj", dest="export_adj", action="store_true")
    p.add_argument("--no-export-adj", dest="export_adj", action="store_false")
    p.set_defaults(export_adj=None)
    p.add_argument("--export-every", type=int, default=None)

    p.add_argument("--cfot-affinity", type=str, default=None, choices=["euclid","cosine","learned"])
    p.add_argument("--cfot-euclid-scale", type=float, default=None)
    p.add_argument("--cfot-cosine-eps", type=float, default=None)
    p.add_argument("--cfot-pos-weight", type=float, default=None)
    p.add_argument("--cfot-vel-weight", type=float, default=None)

    # Per-class viz cadence
    p.add_argument("--class-viz-every", type=int, default=None)
    p.add_argument("--class-viz-max-per-class", type=int, default=None)
    p.add_argument("--track-class", type=int, default=None)
    p.add_argument(
    "--cfot-inject",
    type=str,
    default=None,
    choices=["replace_all", "pre", "after1", "replace_first"],
    help="Where/how to apply CFOT. Use 'replace_all' to replace every 1×k temporal conv."
)

    args = p.parse_args()

    # --- load & override config ---
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    data_root = Path(data_cfg["root"])

    data_dir = data_root / data_cfg["landmarks_dir"]
    ann_train = data_root / data_cfg["annotations"]["train"]
    ann_val   = data_root / data_cfg["annotations"]["val"]

    cfg["data_dir"]  = str(data_dir)
    cfg["ann_train"] = str(ann_train)
    cfg["ann_val"]   = str(ann_val)
    cfg = override(cfg,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed if args.split_seed is not None else cfg.get("split_seed", 42),
        max_T=args.max_T,
        normalize=cfg.get("normalize", False) if args.normalize is None else bool(args.normalize),
        feat=args.feat if args.feat is not None else cfg.get("feat", "xyz+vel"),
        num_class=args.num_class,
        in_ch=args.in_ch,
        dropout=args.dropout,
        graph_strategy=args.graph_strategy,
        max_hop=args.max_hop,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience if args.early_stop_patience is not None else cfg.get("early_stop_patience", 10),
        batch=args.batch,
        lr=args.lr,
        wd=args.wd,
        label_smoothing=args.label_smoothing,
        seed=args.seed if args.seed is not None else cfg.get("seed", 42),
        device=args.device,
        save_dir=args.save_dir if args.save_dir is not None else cfg.get("save_dir", "runs/briareo_stgcn"),
        exp_name=args.exp_name if args.exp_name is not None else cfg.get("exp_name", "stgcn_baseline_xyzvel_aug_crop"),
        aug=(args.aug.lower() == "true") if isinstance(args.aug, str) else cfg.get("aug", False),
        temporal_mode=args.temporal_mode if args.temporal_mode is not None else cfg.get("temporal_mode","crop_repeat"),
        model=args.model if args.model is not None else cfg.get("model", "stgcn"),
        enable_cfot=(cfg.get("enable_cfot", False) if args.enable_cfot is None else bool(args.enable_cfot)),
        cfot_start_epoch=args.cfot_start_epoch if args.cfot_start_epoch is not None else cfg.get("cfot_start_epoch", 8),
        cfot_warmup_epochs=args.cfot_warmup_epochs if args.cfot_warmup_epochs is not None else cfg.get("cfot_warmup_epochs", 12),
        export_adj=(cfg.get("export_adj", True) if args.export_adj is None else bool(args.export_adj)),
        export_every=args.export_every if args.export_every is not None else cfg.get("export_every", 10),
        class_viz_every=args.class_viz_every if args.class_viz_every is not None else cfg.get("class_viz_every", 0),
        class_viz_max_per_class=args.class_viz_max_per_class if args.class_viz_max_per_class is not None else cfg.get("class_viz_max_per_class", 10),
        track_class=args.track_class if args.track_class is not None else cfg.get("track_class", None),
        cfot_inject=args.cfot_inject if args.cfot_inject is not None else cfg.get("cfot_inject", "replace_all"),
        cfot_affinity=args.cfot_affinity if args.cfot_affinity is not None else cfg.get("cfot_affinity","euclid"),
        cfot_euclid_scale=args.cfot_euclid_scale if args.cfot_euclid_scale is not None else cfg.get("cfot_euclid_scale",1.0),
        cfot_cosine_eps=args.cfot_cosine_eps if args.cfot_cosine_eps is not None else cfg.get("cfot_cosine_eps",1e-6),
        cfot_pos_weight=args.cfot_pos_weight if args.cfot_pos_weight is not None else cfg.get("cfot_pos_weight",1.0),
        cfot_vel_weight=args.cfot_vel_weight if args.cfot_vel_weight is not None else cfg.get("cfot_vel_weight",0.2),

    )

    # --- reproducibility & device ---
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # --- IO / logging ---
    save_root = Path(cfg.get("save_dir", "runs/briareo_stgcn"))
    exp_name  = cfg.get("exp_name", "stgcn_baseline_xyzvel_aug_crop")
    workdir   = save_root / exp_name
    workdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(workdir / "tf"))

    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "train_val_metrics.jsonl"

    viz_dir = workdir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # --- make or reuse split ---
    ann_train_path = Path(cfg["ann_train"])
    ann_val_cfg = cfg.get("ann_val", None)
    split_dir = workdir / "splits"
    split_train = split_dir / "train_split.txt"
    split_val   = split_dir / "val_split.txt"

    if cfg.get("val_ratio") is not None:
        items = _read_ann_lines(str(ann_train_path))
        ratio = float(cfg.get("val_ratio", 0.2))
        split_seed = int(cfg.get("split_seed", 42))
        tr_items, va_items = stratified_split(items, ratio, split_seed)
        write_ann_file(split_train, tr_items)
        write_ann_file(split_val, va_items)
        ann_train_used = str(split_train)
        ann_val_used   = str(split_val)
        print(f"[split] Created stratified split (split_seed={split_seed}): train={len(tr_items)} | val={len(va_items)} at {split_dir}")
    else:
        ann_train_used = str(ann_train_path)
        ann_val_used   = str(ann_val_cfg)
        print(f"[split] Using provided train/val lists: {ann_train_used} | {ann_val_used}")

    # --- datasets & loaders ---
    data_dir = cfg.get("data_dir")
    max_T = int(cfg.get("max_T", 180))
    normalize = bool(cfg.get("normalize", False))
    feat = cfg.get("feat", "xyz+vel")

    label_map = build_briareo_label_map(cfg["ann_train"], cfg.get("data_dir"))

    train_ds = BriareoDataset(
        ann_file=ann_train_used, data_dir=data_dir, max_T=max_T, normalize=normalize,
        feat=feat, temporal_mode=cfg.get("temporal_mode", "crop_repeat"),
        aug=bool(cfg.get("aug", False)), eval_mode=False, label_map=label_map
    )
    val_ds = BriareoDataset(
        ann_file=ann_val_used, data_dir=data_dir, max_T=max_T, normalize=normalize,
        feat=feat, temporal_mode=cfg.get("temporal_mode", "crop_repeat"),
        aug=False, eval_mode=True, label_map=label_map
    )
    num_classes = train_ds.num_classes
    cfg["num_class"] = num_classes

    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch", 64)), shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=briareo_collate)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch", 64)), shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=briareo_collate)

    # --- viz logger ---
    global viz_logger
    if bool(cfg.get("export_adj", True)):
        deltas_list: List[int] = cfg.get("cfot_deltas", [1])
        try:
            viz_logger = AdjLogger(V=21, deltas=deltas_list, save_every=int(cfg.get("export_every", 10)))
            print(f"[viz] AdjLogger initialized with V=21, deltas={deltas_list}")
        except Exception as _e:
            print(f"[viz] AdjLogger init failed: {_e}")
            viz_logger = None

    # --- model ---
    model = build_model(cfg, device)
    requested = bool(cfg.get("enable_cfot", False))
    present   = hasattr(model, "cfot_module") and (getattr(model, "cfot_module", None) is not None)
    if cfg.get("enable_cfot", False):
        has = hasattr(model, "cfot_module") and (getattr(model, "cfot_module", None) is not None)
        if not has:
            raise RuntimeError("[CFOT] enable_cfot=True but model has no cfot_module.")
    if requested and not present:
        print("[cfot][WARN] enable_cfot=True but the model has no `cfot_module`.")
    if present:
        n_cfot_params = sum(p.numel() for p in model.cfot_module.parameters())
        print(f"[cfot] Model has CFOT module: params={n_cfot_params} "
              f"| beta={getattr(model, 'cfot_beta', 1.0)} "
              f"| initial_strength={getattr(model, 'cfot_strength_frac', 0.0)}")
    else:
        print("[cfot] Model has NO CFOT module (baseline).")

    # --- epoch-0 viz snapshot ---
    # if viz_logger is not None:
    #     sp0 = None
    #     if hasattr(model, "edge_importance"):
    #         try:
    #             with torch.no_grad():
    #                 _sp = [w.detach().float().cpu().numpy() for w in model.edge_importance]
    #             if len(_sp) > 0:
    #                 sp0 = sum(_sp) / float(len(_sp))
    #         except Exception as _e:
    #             print(f"[viz] spatial epoch0 snapshot skipped: {_e}")

    #     tmp0 = None
    #     tbd0 = None
    #     try:
    #         # run a single forward to populate CFOT buffers
    #         batch0 = next(iter(train_loader))
    #         x0 = batch0["data"][:1].to(device)
    #         with torch.no_grad():
    #             _ = model(x0)
    #         avg_TVV, by_delta = _extract_temporal_from_model(model, V=21)
    #         if avg_TVV is not None:
    #             if avg_TVV.ndim == 3:
    #                 tmp0 = avg_TVV.mean(axis=0)        # [V,V]
    #             elif avg_TVV.ndim == 2:
    #                 tmp0 = avg_TVV
    #         if by_delta is not None:
    #             D0 = len(getattr(viz_logger, "deltas", []) or [])
    #             D0 = max(D0, 1)
    #             tbd0 = _ensure_DVV(by_delta, D=D0, V=21)
    #     except Exception as _e:
    #         print(f"[viz] temporal epoch0 warmup skipped: {_e}")

    #     try:
    #         viz_logger.snapshot_epoch0(spatial=sp0, temporal=tmp0, temporal_by_delta=tbd0, out_dir=viz_dir)
    #         print("[viz] Saved epoch0 snapshots (spatial/temporal).")
    #     except Exception as _e:
    #         print(f"[viz] epoch0 snapshot save failed: {_e}")

    # # optional classwise Δ=1 heatmaps at epoch 0
    # try:
    #     export_delta1_random_frame_heatmaps(
    #         dataset=train_ds, model=model, device=device,
    #         out_dir=workdir / "viz_track", epoch=0,
    #         max_per_class=int(cfg.get("class_viz_max_per_class", 1)),
    #         track_class=cfg.get("track_class")
    #     )
    #     print(f"[viz] Saved Δ=1 random-frame heatmaps @ epoch 0 -> {workdir/'viz_track'}")
    # except Exception as e:
    #     print(f"[viz] epoch0 Δ=1 export failed: {e}")

    # --- optim/loss/schedule ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("wd", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.05)))

    train_cfg = cfg.get("train", {})
    opt_cfg   = cfg.get("opt", {})
    epochs   = int(cfg.get("epochs", train_cfg.get("epochs", 70)))
    base_lr  = float(cfg.get("lr", opt_cfg.get("lr", 1e-3)))
    final_lr = 1e-6
    warmup_epochs = max(1, int(0.05 * epochs))
    sched_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    sched_cosine  = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=final_lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.3, patience=3, cooldown=0, min_lr=1e-6
    # )
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])

    # snapshot config used
    with open(workdir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # --- CFOT schedule ---
    cfot_start = int(cfg.get("cfot_start_epoch", 8))
    cfot_warm  = int(cfg.get("cfot_warmup_epochs", 12))
    cfot_frac_fn = make_cfot_frac_fn(cfot_start, cfot_warm)

    # --- train loop ---
    best_loss = float("inf")
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = int(cfg.get("early_stop_patience", 10))
    epochs = int(cfg.get("epochs", 70))
    num_classes = train_ds.num_classes

    best_path = workdir / "best.pt"
    last_path = workdir / "last.pt"

    print("ARGUMENTS....")
    for k, v in cfg.items():
        print(f"> {k}: {v}")

    train_wall_start = time.perf_counter()

    for epoch in range(1, epochs + 1):

        if viz_logger is not None:
            viz_logger.begin_epoch()

        # CFOT strength schedule
        frac = cfot_frac_fn(epoch)
        if hasattr(model, "set_cfot_strength"):
            model.set_cfot_strength(frac)
        elif hasattr(model, "cfot_module"):
            model.cfot_strength_frac = frac
        print("[chk] strength", float(getattr(model, "cfot_strength_frac", -1)))
        if epoch == cfot_start or epoch == (cfot_start + cfot_warm - 1):
            print(f"[train] CFOT strength frac now {frac:.3f} (epoch {epoch})")

        # optional gate anneal
        if hasattr(model, "cfot_module") and hasattr(model.cfot_module, "set_gumbel_tau"):
            with torch.no_grad():
                prog = (epoch - 1) / max(1, epochs - 1)
                target_tau = 0.5 + 0.5 * (0.5 * (1.0 + math.cos(math.pi * prog)))
                model.cfot_module.set_gumbel_tau(float(target_tau))

        # ---- TRAIN ----
        epoch_train_start = time.perf_counter()
        tr = run_train_epoch(model, train_loader, optimizer, device, criterion,
                             epoch=epoch, writer=writer, do_cfot_check=True, V=21)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        train_epoch_time_s = time.perf_counter() - epoch_train_start
        writer.add_scalar("timing/epoch_train_s", train_epoch_time_s, epoch)
        print(f"[timing] epoch {epoch:03d} train: {train_epoch_time_s:.2f}s")

        # ---- VAL ----
        epoch_val_start = time.perf_counter()
        va = run_val_epoch(model, val_loader, device, criterion,
                           num_classes=num_classes, epoch=epoch, writer=writer)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        val_epoch_time_s = time.perf_counter() - epoch_val_start
        writer.add_scalar("timing/epoch_val_s", val_epoch_time_s, epoch)
        print(f"[timing] epoch {epoch:03d} val:   {val_epoch_time_s:.2f}s")

        scheduler.step()

        # Plateau step
        val_loss = float(va["loss"])
        if not math.isfinite(val_loss):
            val_loss = float("inf")
        # scheduler.step(val_loss)

        # Scalars
        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("loss/train", tr["loss"], epoch)
        writer.add_scalar("acc/train", tr["acc"], epoch)
        writer.add_scalar("loss/val", va["loss"], epoch)
        writer.add_scalar("acc/val", va["acc"], epoch)
        writer.add_scalar("lr", lr_now, epoch)

        print(f"Epoch {epoch:03d}/{epochs}: "
              f"train loss {tr['loss']:.4f} acc {tr['acc']*100:5.2f}% | "
              f"val loss {va['loss']:.4f} acc {va['acc']*100:5.2f}% "
              f"| top1 {va['top1']:.2f}% top5 {va['top5']:.2f}% | lr {lr_now:.2e}")

        # --- VIZ: save epoch snapshots ---
        # if viz_logger is not None:
        #     try:
        #         N = int(cfg.get("export_every", 10))
        #         if (epoch % N == 0) or (epoch == epochs):
        #             viz_logger.end_epoch(epoch, out_dir=viz_dir)  # saves NPY + PNG
        #             print(f"[viz] Saved adjacency snapshots @ epoch {epoch} -> {viz_dir}")
        #     except Exception as _e:
        #         print(f"[viz] end_epoch failed: {_e}")
        # if (epoch % 5) == 0:
        #     try:
        #         export_progression_pairs(
        #             dataset=train_ds,
        #             model=model,
        #             device=device,
        #             out_root=workdir / "viz_pairs",
        #             epoch=epoch,
        #             pairs=(0, 9, 19, 29, 39, 49, 59, 69, 79),  # 0-based => 1–2,10–11,...,80–81
        #             delta_index=0,
        #             one_sample_per_class=True
        #         )
        #         print(f"[viz] Saved per-class progression pairs @ epoch {epoch} -> {workdir/'viz_pairs'/f'ep_{epoch:03d}'}")
        #     except Exception as e:
        #         print(f"[viz] progression export failed @ epoch {epoch}: {e}")

        # --- classwise Δ=1 heatmaps cadence ---
        # cls_every = int(cfg.get("class_viz_every", 0))
        # if (cls_every and (epoch % cls_every == 0)) or (epoch == epochs):
        #     try:
        #         export_delta1_random_frame_heatmaps(
        #             dataset=train_ds, model=model, device=device,
        #             out_dir=workdir / "viz_track", epoch=epoch,
        #             max_per_class=int(cfg.get("class_viz_max_per_class", 1)),
        #             track_class=cfg.get("track_class")
        #         )
        #         print(f"[viz] Saved Δ=1 random-frame heatmaps @ epoch {epoch} -> {workdir/'viz_track'}")
        #     except Exception as e:
        #         print(f"[viz] Δ=1 export failed @ epoch {epoch}: {e}")

        # ---- metrics & cm ----
        cfot_tau_val = None
        cfot_topk_val = None
        if hasattr(model, "cfot_module") and (model.cfot_module is not None):
            if hasattr(model.cfot_module, "tau"):
                t = getattr(model.cfot_module, "tau")
                cfot_tau_val = float(t) if not isinstance(t, torch.Tensor) else float(t.detach().cpu().item())
            elif hasattr(model.cfot_module, "gate_tau"):
                t = getattr(model.cfot_module, "gate_tau")
                cfot_tau_val = float(t) if not isinstance(t, torch.Tensor) else float(t.detach().cpu().item())
            if hasattr(model.cfot_module, "topk"):
                cfot_topk_val = int(getattr(model.cfot_module, "topk"))

        cm = va["cm"]
        per_class_recall = [float(r) for r in va["per_class_recall"]]

        write_jsonl(metrics_path, {
            "epoch": int(epoch),
            "split": "train",
            "loss": float(tr["loss"]),
            "acc": float(tr["acc"]),
            "epoch_time_s": float(train_epoch_time_s),
        })
        write_jsonl(metrics_path, {
            "epoch": int(epoch),
            "split": "val",
            "loss": float(va["loss"]),
            "acc": float(va["acc"]),
            "top1": float(va["top1"]),
            "top5": float(va["top5"]),
            "per_class_recall": per_class_recall,
            "epoch_time_s": float(val_epoch_time_s),
            "cfot_tau": cfot_tau_val,
            "cfot_topk": cfot_topk_val,
        })

        cm_path = logs_dir / f"val_confmat_epoch{epoch:03d}.csv"
        save_numpy_csv(cm_path, cm, header="rows=true_label, cols=pred_label")

        # ---- checkpoints ----
        ckpt_common = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": cfg,
            "val_acc": va["acc"],
            "val_loss": va["loss"],
            "cfot_strength_frac": float(getattr(model, "cfot_strength_frac", 1.0)),
        }
        torch.save(ckpt_common, last_path)

        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_acc = va["acc"]
            best_epoch = epoch
            torch.save(ckpt_common, best_path)
            print(f" Validation loss improved to {best_loss:.5f}. Saved: {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" Early Stopping: no val-loss improvement for {patience} epochs "
                      f"(best at epoch {best_epoch}: {best_loss:.5f}).")
                break

    total_train_time_s = time.perf_counter() - train_wall_start
    print(f"[timing] total training wall time: {total_train_time_s:.2f}s")
    write_jsonl(logs_dir / "training_summary.jsonl", {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "best_val_acc": float(best_acc),
        "total_train_wall_s": float(total_train_time_s),
        "epochs_run": int(epoch),
    })

    # final classwise export if modulo skipped
    # try:
    #     if (epoch % max(1, int(cfg.get("class_viz_every", 5)))) != 0:
    #         export_delta1_random_frame_heatmaps(
    #             dataset=train_ds, model=model, device=device,
    #             out_dir=workdir / "viz_track", epoch=epoch,
    #             max_per_class=int(cfg.get("class_viz_max_per_class", 1)),
    #             track_class=cfg.get("track_class"),
    #         )
    #         print(f"[viz] Saved Δ=1 random-frame heatmaps @ final epoch {epoch} -> {workdir/'viz_track'}")
    # except Exception as e:
    #     print(f"[viz] final Δ=1 export failed: {e}")

    print(f"Done. Best @ epoch {best_epoch}: val_loss={best_loss:.5f}, best_val_acc={best_acc*100:5.2f}%.")

if __name__ == "__main__":
    main()
