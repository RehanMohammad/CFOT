#!/usr/bin/env python3
# ST-GCN / MSG3D training on IPN with optional CFOT + run header + profiling
#!/usr/bin/env python3
# ST-GCN / MSG3D training on IPN with optional CFOT + detailed timing

from __future__ import annotations

import os, math, random, inspect, time, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence, Union

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

try:
    from torchviz import make_dot
except ImportError:
    make_dot = None

# Project modules
from dataset.ipn import IPNDataset, build_ipn_label_map, ipn_collate

# ------------------------------
# Reproducibility
# ------------------------------
def force_reproducible(seed: int = 42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# For dataloader reproducibility
_g = torch.Generator()
_g.manual_seed(42)
def _winit(worker_id: int):
    wseed = 42 + worker_id
    np.random.seed(wseed)
    random.seed(wseed)

# ------------------------------
# Small utils
# ------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def override(cfg: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    for k, v in kwargs.items():
        if v is not None: cfg[k] = v
    return cfg

def write_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def save_numpy_csv(path: Path, array: np.ndarray, header: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), array, delimiter=",", fmt="%d", header=header)

def accuracy_simple(logits: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return float((pred == target).sum().item()) / float(target.numel())

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
    try:
        from sklearn.metrics import classification_report
    except Exception:
        classification_report = None
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes: cm[t, p] += 1
    report = ""
    if classification_report is not None:
        try: report = classification_report(y_true, y_pred, digits=4)
        except Exception: pass
    return cm, report

def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def try_flops_g(model: nn.Module, in_shape: Tuple[int,int,int], device) -> Optional[float]:
    C, T, V = in_shape
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.zeros((1, C, T, V, 1), device=device)
        flops = FlopCountAnalysis(model.eval(), x).total()
        return float(flops) / 1e9
    except Exception:
        pass
    try:
        from ptflops import get_model_complexity_info
        class Wrap5D(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x3): return self.m(x3.unsqueeze(-1))
        w = Wrap5D(model.eval()).to(device)
        macs, _ = get_model_complexity_info(
            w, (C, T, V),
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return float(macs) * 2.0 / 1e9
    except Exception:
        return None

@torch.no_grad()
def profile_inference(model: nn.Module, device, C: int, T: int, V: int,
                      warm=30, iters=200, batch1_only=True, batch_size: int = 64):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    x1 = torch.zeros((1, C, T, V, 1), device=device)
    for _ in range(warm): _ = model(x1)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): _ = model(x1)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()
    latency_ms = 1000.0 * (t1 - t0) / iters
    thr = None
    if not batch1_only:
        bs = int(batch_size)
        xb = torch.zeros((bs, C, T, V, 1), device=device)
        for _ in range(warm): _ = model(xb)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        s0 = time.perf_counter()
        for _ in range(iters): _ = model(xb)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        s1 = time.perf_counter()
        thr = (iters * bs) / (s1 - s0)
    return {"latency_ms_b1": float(latency_ms),
            "throughput_clips_s": None if batch1_only else float(thr)}

def env_info():
    gname = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "gpu": gname,
        "cuda": torch.version.cuda if hasattr(torch.version, "cuda") else "NA",
        "cudnn": torch.backends.cudnn.version(),
        "torch": torch.__version__,
        "amp": False,
    }

def print_kv_table(title: str, kv: Dict[str, Any]):
    print(f"\n[{title}]")
    for k in sorted(kv.keys()):
        print(f"{k:>24}: {kv[k]}")

# ------------------------------
# Split helpers
# ------------------------------
def _read_ann_lines(ann_path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'): continue
            parts = s.split()
            if len(parts) < 2: raise ValueError(f"Bad line in {ann_path}: '{s}'")
            path_part = " ".join(parts[:-1]); label = int(parts[-1])
            items.append((path_part, label))
    if not items: raise ValueError(f"No items found in {ann_path}")
    return items

def stratified_split(items, val_ratio: float, seed: int):
    by_label: Dict[int, List[str]] = {}
    for p, y in items: by_label.setdefault(y, []).append(p)
    rng = random.Random(seed)
    train_items, val_items = [], []
    for y, paths in by_label.items():
        rng.shuffle(paths); n = len(paths)
        nv = 0 if n == 1 else max(1, min(n - 1, int(round(n * val_ratio))))
        val_paths = paths[:nv]; tr_paths = paths[nv:]
        val_items.extend([(p, y) for p in val_paths])
        train_items.extend([(p, y) for p in tr_paths])
    rng.shuffle(train_items); rng.shuffle(val_items)
    return train_items, val_items

def write_ann_file(path: Path, items: List[Tuple[str,int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, y in items: f.write(f"{p} {y}\n")

# ------------------------------
# CFOT schedule
# ------------------------------
def make_cfot_frac_fn(start: int, warm: int):
    def f(epoch_1based: int) -> float:
        if epoch_1based < start: return 0.0
        if warm <= 0: return 1.0
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
    feat = cfg.get("feat", "xyz")
    if feat == "xyz":
        in_ch = 3 
    from models.stgcn.stgcn import Model as STGCN
    base_kwargs = dict(
        in_channels=in_ch,
        num_class=int(cfg.get("num_class", 14)),
        graph_args={"layout":"ipn",
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
        cfot_max_k=int(cfg.get("cfot_max_k", 21)),
        cfot_affinity=cfg.get("cfot_affinity","euclid"),
        cfot_euclid_scale=float(cfg.get("cfot_euclid_scale",1.0)),
        cfot_cosine_eps=float(cfg.get("cfot_cosine_eps",1e-6)),
        cfot_pos_weight=float(cfg.get("cfot_pos_weight",1.0)),
        cfot_vel_weight=float(cfg.get("cfot_vel_weight",0.2)),
    )
    return _safe_build(STGCN, base_kwargs, maybe_kwargs).to(device)

# ------------------------------
# OFFLINE AUGMENTATION
# ------------------------------
def _seed_scope(seed: int):
    class _Scope:
        def __enter__(self):
            self.state_py = random.getstate()
            self.state_np = np.random.get_state()
            random.seed(seed); np.random.seed(seed)
        def __exit__(self, exc_type, exc, tb):
            random.setstate(self.state_py)
            np.random.set_state(self.state_np)
    return _Scope()

def _tensor_to_xyz(data: torch.Tensor) -> torch.Tensor:
    # data: [C,T,V,1], return xyz [3,T,V,1]
    C = data.shape[0]
    if C >= 3:
        return data[:3]
    raise ValueError("Expected at least 3 channels for xyz")

def _xyz_to_feat(xyz: torch.Tensor, want_feat: str) -> torch.Tensor:
    # xyz: [3,T,V,1] -> feat
    if want_feat == "xyz":
        return xyz
    # xyz+vel: compute first-order difference
    x = xyz
    vel = torch.zeros_like(x)
    vel[:,1:] = x[:,1:] - x[:,:-1]
    return torch.cat([x, vel], dim=0)  # [6,T,V,1]

def _apply_simple_aug(xyz: torch.Tensor) -> torch.Tensor:
    # xyz: [3,T,V,1]
    x = xyz.clone()
    # scale
    s = float(np.random.uniform(0.9, 1.1))
    x = x * s
    # small rotation around z (in-plane) in radians
    theta = float(np.random.uniform(-np.pi/12, np.pi/12))
    c, si = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -si, 0.0],
                      [si,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=x.dtype, device=x.device)
    # x is [3,T,V,1] -> reshape to [3, T*V]
    tv = x.shape[1]*x.shape[2]
    xr = x.view(3, tv)
    xr = R @ xr
    x = xr.view(3, x.shape[1], x.shape[2], 1)
    # jitter
    noise = torch.randn_like(x) * 0.01
    x = x + noise
    return x

def build_offline_cache(
    base_train_ds: IPNDataset,
    factor: int,
    cache_dir: Path,
    feat: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Writes augmented copies to disk as .pt files: cache_dir/copy_k/{index}.pt
    Each file stores a dict with keys: data [C,T,V,1] float16, label int64, meta dict.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_len": len(base_train_ds),
        "factor": int(factor),
        "feat": str(feat),
        "version": 1,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Iterate each augmented copy
    for k in range(factor):
        copy_dir = cache_dir / f"copy_{k}"
        copy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[offline] building copy {k+1}/{factor} -> {copy_dir}")
        with _seed_scope(seed + 1000 * (k+1)):
            for i in range(len(base_train_ds)):
                sample = base_train_ds[i]  # aug=False in base
                data = sample["data"]      # [C,T,V,1]
                label = int(sample["label"])
                meta_i = dict(sample["meta"])
                # move to CPU tensor float32
                d = data.detach().to(dtype=torch.float32, device="cpu")
                xyz = _tensor_to_xyz(d)                      # [3,T,V,1]
                xyz_aug = _apply_simple_aug(xyz)             # [3,T,V,1]
                d_aug = _xyz_to_feat(xyz_aug, feat)          # [C or 6,T,V,1]
                rec = {"data": d_aug.to(torch.float16), "label": label, "meta": meta_i}
                torch.save(rec, copy_dir / f"{i}.pt")
                if (i+1) % 500 == 0 or (i+1) == len(base_train_ds):
                    print(f"  [{k+1}/{factor}] cached {i+1}/{len(base_train_ds)}")
    return meta

class OfflineAugDataset(Dataset):
    """
    Wraps a base dataset and appends K offline-augmented copies stored on disk.
    Length = base_len * (1 + K).
    """
    def __init__(self, base_ds: IPNDataset, cache_dir: Path, factor: int):
        self.base = base_ds
        self.base_len = len(base_ds)
        self.cache_dir = cache_dir
        self.factor = int(factor)
        self.copy_dirs = [cache_dir / f"copy_{k}" for k in range(factor)]
        for d in self.copy_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Missing cache copy directory: {d}")
        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["base_len"] == self.base_len, "cache/base length mismatch"
        assert meta["factor"]   == self.factor,   "cache/factor mismatch"

    def __len__(self) -> int:
        return self.base_len * (1 + self.factor)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self.base_len:
            return self.base[idx]
        # augmented part
        j = idx - self.base_len
        k = j // self.base_len     # which augmented copy
        i = j %  self.base_len     # base index
        rec = torch.load(self.copy_dirs[k] / f"{i}.pt", map_location="cpu")
        # ensure output types align with IPNDataset
        out = {
            "data": rec["data"].to(dtype=torch.float32),  # cast back to fp32
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "meta": dict(rec.get("meta", {})),
        }
        return out

# ------------------------------
# Epochs with detailed timing
# ------------------------------
def _cuda_event_pair():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None

def run_train_epoch(model: nn.Module, loader: DataLoader, optimizer, device, criterion,
                    epoch: Optional[int] = None, writer: Optional[SummaryWriter] = None) -> Dict[str, float]:
    model.train(True)
    tot_loss = 0.0; tot_acc = 0.0; tot_n = 0
    data_times: List[float] = []
    step_times_ms: List[float] = []

    end = time.perf_counter()
    ev_start, ev_end = _cuda_event_pair()

    for batch in loader:
        data_times.append(time.perf_counter() - end)

        x = batch["data"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if ev_start is not None and ev_end is not None:
            ev_start.record()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        cls_loss = criterion(logits, y)
        reg = 0.0
        if hasattr(model, "cfot_module") and hasattr(model.cfot_module, "regularization_loss"):
            reg = model.cfot_module.regularization_loss()
        loss = cls_loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if ev_start is not None and ev_end is not None:
            ev_end.record()
            torch.cuda.synchronize()
            step_times_ms.append(ev_start.elapsed_time(ev_end))
        else:
            step_times_ms.append(0.0)

        bs = y.size(0)
        tot_loss += float(loss.item()) * bs
        tot_acc  += accuracy_simple(logits, y) * bs
        tot_n    += bs

        end = time.perf_counter()

    denom = max(1, tot_n)
    return {
        "loss": tot_loss/denom,
        "acc":  tot_acc/denom,
        "data_ms": 1000.0 * float(np.mean(data_times)) if data_times else 0.0,
        "step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
    }

@torch.no_grad()
def run_val_epoch(model: nn.Module, loader: DataLoader, device, criterion,
                  num_classes: Optional[int],
                  epoch: Optional[int] = None, writer: Optional[SummaryWriter] = None) -> Dict[str, Any]:
    model.train(False)
    tot_loss = tot_acc = 0.0; tot_n = 0
    top1_meter = top5_meter = 0.0
    y_true, y_pred = [], []
    for batch in loader:
        x = batch["data"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
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
        tp = cm[c, c]; fn = cm[c, :].sum() - tp
        per_class_recall.append(float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0)
    if writer is not None and epoch is not None:
        writer.add_scalar("topk/val_top1", top1, epoch)
        writer.add_scalar("topk/val_top5", top5, epoch)
    return {"loss": loss_epoch, "acc": acc_epoch, "top1": top1, "top5": top5,
            "cm": cm, "per_class_recall": per_class_recall, "report": cls_report_txt}

# ------------------------------
# Main
# ------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)

    # IO / data overrides
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--ann-train", type=str, default=None)
    p.add_argument("--ann-val", type=str, default=None)
    p.add_argument("--ann-test", type=str, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--max-T", type=int, default=None)
    p.add_argument("--normalize", dest="normalize", action="store_true")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.set_defaults(normalize=None)
    # model/training
    p.add_argument("--feat", type=str, default=None, choices=["xyz"])
    p.add_argument("--num-class", type=int, default=None)
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
    p.add_argument("--aug", type=str, default=None)
    p.add_argument("--temporal-mode", type=str, default=None, choices=["interp"])
    p.add_argument("--model", type=str, default=None, choices=["stgcn"])
    # CFOT
    p.add_argument("--enable-cfot",  dest="enable_cfot", action="store_true")
    p.add_argument("--disable-cfot", dest="enable_cfot", action="store_false")
    p.set_defaults(enable_cfot=None)
    p.add_argument("--cfot-start-epoch", type=int, default=None)
    p.add_argument("--cfot-warmup-epochs", type=int, default=None)
    p.add_argument("--cfot-affinity", type=str, default=None, choices=["euclid","cosine","learned"])
    p.add_argument("--cfot-euclid-scale", type=float, default=None)
    p.add_argument("--cfot-cosine-eps", type=float, default=None)
    p.add_argument("--cfot-pos-weight", type=float, default=None)
    p.add_argument("--cfot-vel-weight", type=float, default=None)
    p.add_argument("--cfot-inject", type=str, default=None,
                   choices=["pre"])

    # OFFLINE AUG
    p.add_argument("--offline_aug_factor", type=int, default=0)
    p.add_argument("--offline_cache_dir", type=str, default=None)
    p.add_argument("--build_cache", action="store_true")

    # profiling and eval
    p.add_argument("--profile", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--load", type=str, default=None)

    args = p.parse_args()

    # Repro + device
    force_reproducible(42)
    cfg = load_config(args.config)
    cfg = override(cfg,
        data_dir=args.data_dir,
        ann_train=args.ann_train,
        ann_val=args.ann_val,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed if args.split_seed is not None else cfg.get("split_seed", 42),
        max_T=args.max_T,
        normalize=cfg.get("normalize", False) if args.normalize is None else bool(args.normalize),
        feat=args.feat if args.feat is not None else cfg.get("feat", "xyz+vel"),
        num_class=args.num_class,
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
        save_dir=args.save_dir if args.save_dir is not None else cfg.get("save_dir", "runs/ipn_stgcn"),
        exp_name=args.exp_name if args.exp_name is not None else cfg.get("exp_name", "stgcn_baseline_xyzvel"),
        aug=(args.aug.lower() == "true") if isinstance(args.aug, str) else cfg.get("aug", False),
        temporal_mode=args.temporal_mode if args.temporal_mode is not None else cfg.get("temporal_mode","crop_repeat"),
        model=args.model if args.model is not None else cfg.get("model", "stgcn"),
        enable_cfot=(cfg.get("enable_cfot", False) if args.enable_cfot is None else bool(args.enable_cfot)),
        cfot_start_epoch=args.cfot_start_epoch if args.cfot_start_epoch is not None else cfg.get("cfot_start_epoch", 8),
        cfot_warmup_epochs=args.cfot_warmup_epochs if args.cfot_warmup_epochs is not None else cfg.get("cfot_warmup_epochs", 12),
        cfot_inject=args.cfot_inject if args.cfot_inject is not None else cfg.get("cfot_inject", "replace_all"),
        cfot_affinity=args.cfot_affinity if args.cfot_affinity is not None else cfg.get("cfot_affinity","euclid"),
        cfot_euclid_scale=args.cfot_euclid_scale if args.cfot_euclid_scale is not None else cfg.get("cfot_euclid_scale",1.0),
        cfot_cosine_eps=args.cfot_cosine_eps if args.cfot_cosine_eps is not None else cfg.get("cfot_cosine_eps",1e-6),
        cfot_pos_weight=args.cfot_pos_weight if args.cfot_pos_weight is not None else cfg.get("cfot_pos_weight",1.0),
        cfot_vel_weight=args.cfot_vel_weight if args.cfot_vel_weight is not None else cfg.get("cfot_vel_weight",0.2),
    )
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # IO / logging
    save_root = Path(cfg.get("save_dir", "runs/ipn_stgcn"))
    exp_name  = cfg.get("exp_name", "stgcn_baseline_xyzvel")
    workdir   = save_root / exp_name
    workdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(workdir / "tf"))
    logs_dir = workdir / "logs"; logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "train_val_metrics.jsonl"

    # Split
    ann_train_path = Path(cfg["ann_train"])
    ann_val_cfg = cfg.get("ann_val", None)
    split_dir = workdir / "splits"
    split_train = split_dir / "train_split.txt"
    split_val   = split_dir / "val_split.txt"

    if not ann_val_cfg or (isinstance(ann_val_cfg, str) and (ann_val_cfg.strip().lower() in {"", "none", "null"} or not Path(ann_val_cfg).exists())):
        items = _read_ann_lines(str(ann_train_path))
        ratio = float(cfg.get("val_ratio", 0.2)); split_seed = int(cfg.get("split_seed", 42))
        tr_items, va_items = stratified_split(items, ratio, split_seed)
        write_ann_file(split_train, tr_items); write_ann_file(split_val, va_items)
        ann_train_used = str(split_train); ann_val_used = str(split_val)
        print(f"[split] Created stratified split (seed={split_seed}): train={len(tr_items)} | val={len(va_items)}")
    else:
        ann_train_used = str(ann_train_path); ann_val_used = str(ann_val_cfg)
        print(f"[split] Using provided train/val lists: {ann_train_used} | {ann_val_used}")

    # Datasets & loaders
    data_dir = cfg.get("data_dir"); max_T = int(cfg.get("max_T", 180))
    normalize = bool(cfg.get("normalize", False))
    feat_raw = cfg.get("feat", "xyz")
    feat = str(feat_raw).replace(" ", "").lower()
    cfg["feat"] = feat
    ### END NEW

    label_map = build_ipn_label_map(cfg["ann_train"], cfg.get("data_dir"))

    # base training dataset WITHOUT online aug
    train_ds_base = IPNDataset(
        ann_file=ann_train_used, data_dir=data_dir, max_T=max_T, normalize=normalize,
        feat=feat, temporal_mode=cfg.get("temporal_mode", "crop_repeat"),
        aug=False, eval_mode=False, label_map=label_map
    )
    val_ds = IPNDataset(
        ann_file=ann_val_used, data_dir=data_dir, max_T=max_T, normalize=normalize,
        feat=feat, temporal_mode=cfg.get("temporal_mode", "crop_repeat"),
        aug=False, eval_mode=True, label_map=label_map
    )
    num_classes = train_ds_base.num_classes
    cfg["num_class"] = num_classes

    # ----- OFFLINE AUGMENTATION INTEGRATION -----
    offline_factor = int(args.offline_aug_factor) if hasattr(args, "offline_aug_factor") else 0
    cache_dir = Path(args.offline_cache_dir) if args.offline_cache_dir else None

    if offline_factor > 0:
        if cache_dir is None:
            raise ValueError("Provide --offline_cache_dir when using --offline_aug_factor > 0")
        if args.build_cache:
            # build cache from base dataset
            build_offline_cache(train_ds_base, factor=offline_factor, cache_dir=cache_dir, feat=feat, seed=int(cfg.get("seed",42)))
        # wrap with augmented copies (will error if cache missing)
        train_ds = OfflineAugDataset(train_ds_base, cache_dir=cache_dir, factor=offline_factor)
    else:
        # fallback: respect online aug flag in config if desired
        train_ds = train_ds_base if not bool(cfg.get("aug", False)) else IPNDataset(
            ann_file=ann_train_used, data_dir=data_dir, max_T=max_T, normalize=normalize,
            feat=feat, temporal_mode=cfg.get("temporal_mode", "crop_repeat"),
            aug=True, eval_mode=False, label_map=label_map
        )
    # --------------------------------------------

    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch", 64)), shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ipn_collate,
                              worker_init_fn=_winit, generator=_g, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch", 64)), shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=ipn_collate,
                            worker_init_fn=_winit, generator=_g)

    print("train_len", len(train_ds), "batches/epoch", len(train_loader))

    # Model
    model = build_model(cfg, device)
    if cfg.get("enable_cfot", False):
        has = hasattr(model, "cfot_module") and (getattr(model, "cfot_module", None) is not None)
        if not has:
            raise RuntimeError("[CFOT] enable_cfot=True but model has no cfot_module.")
        n_cfot_params = sum(p.numel() for p in model.cfot_module.parameters())
        print(f"[cfot] Enabled. cfot_params={n_cfot_params}")
    else:
        print("[cfot] Disabled.")

    ### NEW: add model graph to TensorBoard (Graphs tab)
        # Model
    model = build_model(cfg, device)
    if cfg.get("enable_cfot", False):
        has = hasattr(model, "cfot_module") and (getattr(model, "cfot_module", None) is not None)
        if not has:
            raise RuntimeError("[CFOT] enable_cfot=True but model has no cfot_module.")
        n_cfot_params = sum(p.numel() for p in model.cfot_module.parameters())
        print(f"[cfot] Enabled. cfot_params={n_cfot_params}")
    else:
        print("[cfot] Disabled.")

    # ------------------------------
    # Graph visualization
    # ------------------------------
    try:
        # Take one batch from the train loader
        example_batch = next(iter(train_loader))
        x_example = example_batch["data"].to(device)

        # If using DataParallel, unwrap
        model_for_graph = model.module if hasattr(model, "module") else model

        # 1) Add graph to TensorBoard
        print("[tb] Adding model graph to TensorBoard...")
        writer.add_graph(model_for_graph, x_example)
        writer.flush()
        print("[tb] Model graph added. View it in TensorBoard -> Graphs.")

        # 2) (Optional) Export graph as PDF using torchviz
        if make_dot is not None:
            print("[torchviz] Building PDF graph...")
            model_for_graph.eval()

            # Forward pass with gradients enabled (no torch.no_grad here)
            y_example = model_for_graph(x_example)

            # torchviz graph (use first element if output is a tuple/list)
            if isinstance(y_example, (list, tuple)):
                y_out = y_example[0]
            else:
                y_out = y_example

            dot = make_dot(
                y_out,
                params=dict(model_for_graph.named_parameters()),
                show_attrs=False,    # smaller graph
                show_saved=False,
            )

            graphs_dir = workdir / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = graphs_dir / "ipn_stgcn_cfot_graph"

            # This generates ipn_stgcn_cfot_graph.pdf
            dot.render(str(pdf_path), format="pdf", cleanup=True)
            print(f"[torchviz] PDF graph saved to: {pdf_path.with_suffix('.pdf')}")
        else:
            print("[torchviz] make_dot is not available (torchviz not installed). Skipping PDF export.")

    except Exception as e:
        print(f"[graph] Skipped graph export due to error: {e}")
    ### END NEW

    # Run header
    C = 3 if cfg.get("feat","xyz+vel") == "xyz" else 6
    T = int(cfg.get("max_T", 180)); V = 21
    hdr_cfg = {
        "model": cfg.get("model","stgcn"),
        "feat": cfg.get("feat","xyz+vel"),
        "T": T, "V": V, "batch": int(cfg.get("batch",64)),
        "epochs": int(cfg.get("epochs",70)),
        "optimizer": "AdamW",
        "lr": cfg.get("lr",1e-3), "wd": cfg.get("wd",1e-4),
        "label_smoothing": cfg.get("label_smoothing",0.05),
        "dropout": cfg.get("dropout",0.05),
        "aug_train": bool(cfg.get("aug", False)),
        "temporal_mode": cfg.get("temporal_mode","crop_repeat"),
        "enable_cfot": bool(cfg.get("enable_cfot", False)),
        "seed": cfg.get("seed",42),
        "offline_aug_factor": offline_factor,
        "offline_cache_dir": str(cache_dir) if cache_dir else None,
    }
    hdr_env = env_info()
    hdr_model = {"params_M": round(count_params_m(model), 3),
                 "flops_G_per_clip": None, "latency_ms_b1": None, "throughput_clips_s": None}
    if args.profile:
        hdr_model["flops_G_per_clip"] = try_flops_g(model, (C,T,V), device)
        prof = profile_inference(model, device, C, T, V, warm=20, iters=100, batch1_only=False)
        hdr_model.update(prof)
    print_kv_table("CONFIG", hdr_cfg)
    print_kv_table("ENV", hdr_env)
    print_kv_table("MODEL", hdr_model)
    with open(workdir / "run_header.json", "w", encoding="utf-8") as f:
        json.dump({"config": hdr_cfg, "env": hdr_env, "model": hdr_model}, f, indent=2)
    write_jsonl(logs_dir / "run_header.jsonl", {"config": hdr_cfg, "env": hdr_env, "model": hdr_model})
    try:
        flat = {}
        for k,v in hdr_cfg.items(): flat[k]=v
        writer.add_hparams(flat, {})
    except Exception:
        pass

    with open(workdir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Optim/loss/schedule
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg.get("lr", 1e-3)),
                                  weight_decay=float(cfg.get("wd", 1e-4)))
    print("LRs per param group:", [g["lr"] for g in optimizer.param_groups])

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.1)))

    train_cfg = cfg.get("train", {})
    opt_cfg   = cfg.get("opt", {})
    epochs   = int(cfg.get("epochs", train_cfg.get("epochs", 70)))
    base_lr  = float(cfg.get("lr", opt_cfg.get("lr", 1e-3)))
    final_lr = 1e-6
    warmup_epochs = max(1, int(0.05 * epochs))
    sched_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    sched_cosine  = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=final_lr)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])

    # CFOT schedule
    cfot_start = int(cfg.get("cfot_start_epoch", 8))
    cfot_warm  = int(cfg.get("cfot_warmup_epochs", 12))
    cfot_frac_fn = make_cfot_frac_fn(cfot_start, cfot_warm)

    # Train loop
    best_loss = float("inf"); best_acc = 0.0; best_epoch = 0
    epochs_no_improve = 0; patience = int(cfg.get("early_stop_patience", 10))
    best_path = workdir / "best.pt"; last_path = workdir / "last.pt"

    train_wall_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        frac = cfot_frac_fn(epoch)
        if hasattr(model, "set_cfot_strength"): model.set_cfot_strength(frac)
        elif hasattr(model, "cfot_module"): model.cfot_strength_frac = frac
        if epoch == cfot_start or epoch == (cfot_start + cfot_warm - 1):
            print(f"[train] CFOT strength frac {frac:.3f} @ epoch {epoch}")

        # TRAIN
        epoch_train_start = time.perf_counter()
        tr = run_train_epoch(model, train_loader, optimizer, device, criterion, epoch=epoch, writer=writer)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        train_epoch_time_s = time.perf_counter() - epoch_train_start
        writer.add_scalar("timing/epoch_train_s", train_epoch_time_s, epoch)
        writer.add_scalar("timing/avg_batch_data_ms", tr["data_ms"], epoch)
        writer.add_scalar("timing/avg_batch_step_ms", tr["step_ms"], epoch)
        print(f"[timing] epoch {epoch:03d} train: {train_epoch_time_s:.2f}s | "
              f"data {tr['data_ms']:.2f} ms/batch | step {tr['step_ms']:.2f} ms/batch")

        # VAL
        epoch_val_start = time.perf_counter()
        va = run_val_epoch(model, val_loader, device, criterion, num_classes=num_classes, epoch=epoch, writer=writer)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        val_epoch_time_s = time.perf_counter() - epoch_val_start
        writer.add_scalar("timing/epoch_val_s", val_epoch_time_s, epoch)
        print(f"[timing] epoch {epoch:03d} val:   {val_epoch_time_s:.2f}s")

        # Scheduler
        scheduler.step()

        # Scalars
        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("loss/train", tr["loss"], epoch)
        writer.add_scalar("acc/train", tr["acc"], epoch)
        writer.add_scalar("loss/val", va["loss"], epoch)
        writer.add_scalar("acc/val", va["acc"], epoch)
        writer.add_scalar("lr", lr_now, epoch)

        print(f"Epoch {epoch:03d}/{epochs}: "
              f"train loss {tr['loss']:.4f} acc {tr['acc']*100:5.2f}% | "
              f"val loss {va['loss']:.4f} acc {va['acc']*100:5.2f}% | "
              f"top1 {va['top1']:.2f}% top5 {va['top5']:.2f}% | lr {lr_now:.2e}")

        # Metrics rows
        write_jsonl(metrics_path, {
            "epoch": int(epoch), "split": "train",
            "loss": float(tr["loss"]), "acc": float(tr["acc"]),
            "epoch_time_s": float(train_epoch_time_s),
            "avg_batch_data_ms": float(tr["data_ms"]),
            "avg_batch_step_ms": float(tr["step_ms"]),
        })
        write_jsonl(metrics_path, {
            "epoch": int(epoch), "split": "val",
            "loss": float(va["loss"]), "acc": float(va["acc"]),
            "top1": float(va["top1"]), "top5": float(va["top5"]),
            "per_class_recall": [float(r) for r in va["per_class_recall"]],
            "epoch_time_s": float(val_epoch_time_s),
        })

        # Confusion matrix snapshot
        cm_path = logs_dir / f"val_confmat_epoch{epoch:03d}.csv"
        save_numpy_csv(cm_path, va["cm"], header="rows=true_label, cols=pred_label")

        # Checkpoints
        ckpt_common = {"epoch": epoch, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(),
                       "cfg": cfg, "val_acc": va["acc"], "val_loss": va["loss"],
                       "cfot_strength_frac": float(getattr(model, "cfot_strength_frac", 1.0))}
        torch.save(ckpt_common, last_path)
        if float(va["loss"]) + 1e-8 < best_loss:
            best_loss = float(va["loss"]); best_acc = float(va["acc"]); best_epoch = epoch
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
    print(f"Done. Best @ epoch {best_epoch}: val_loss={best_loss:.5f}, best_val_acc={best_acc*100:5.2f}%.")

if __name__ == "__main__":
    main()

