# train_indian_sign.py
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
except Exception:
    make_dot = None

# Import the dataset module you saved earlier (adjust the module name/path if needed)
# Example: place `indian_sign_dataset_v2.py` next to this train script.
from dataset.isl import IndianSignSequenceDataset, collate_sign, build_items_from_split

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
# Small utils (as in your template)
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
# Split helpers (adapted)
# ------------------------------
def stratified_split(items: List[Tuple[str,int]], val_ratio: float, seed: int):
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
# (This reuses the template's builder. It expects model sources in models/ package.)
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

    # Keep same model selection logic as in your template (assumes models/ exists)
    if model_name in ("msg3d_2s","twostreammsg3d","msg3d_two_stream","2s_msg3d"):
        from models.MSG3D.msg3d_two_stream import TwoStreamMSG3D
        model = TwoStreamMSG3D(
            num_class=int(cfg.get("num_class",14)),
            num_point=22, num_person=1,
            graph="utils.graph.Graph",
            graph_args={"layout":"isl",
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
            graph_args={"layout":"isl",
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
            graph_args={"layout":"isl",
                        "strategy":cfg.get("graph_strategy","spatial"),
                        "max_hop": int(cfg.get("max_hop",1))},
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.10)),
        ).to(device)

    # default: ST-GCN
    from models.stgcn.stgcn import Model as STGCN
    base_kwargs = dict(
        in_channels=in_ch,
        num_class=int(cfg.get("num_class", 50)),
        graph_args={"layout":"isl",
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
# Epochs with detailed timing (copied from template)
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

    # IO / data overrides (use directories: train_dir / val_dir / test_dir)
    p.add_argument("--train-dir", type=str, default=None)
    p.add_argument("--val-dir", type=str, default=None)
    p.add_argument("--test-dir", type=str, default=None)
    p.add_argument("--max-T", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--offline_aug_factor", type=int, default=0)
    p.add_argument("--offline_cache_dir", type=str, default=None)
    p.add_argument("--build_cache", action="store_true")

    # model/training
    p.add_argument("--feat", type=str, default=None, choices=["xyz","xyz+vel"])
    p.add_argument("--num-class", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--graph-strategy", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--model", type=str, default=None)

    # profiling and eval
    p.add_argument("--profile", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--load", type=str, default=None)

    args = p.parse_args()

    # Repro + device
    force_reproducible(42)
    cfg = load_config(args.config)
    cfg = override(cfg,
        train_dir=args.train_dir if args.train_dir is not None else cfg.get("train_dir"),
        val_dir=args.val_dir if args.val_dir is not None else cfg.get("val_dir"),
        test_dir=args.test_dir if args.test_dir is not None else cfg.get("test_dir"),
        max_T=args.max_T,
        feat=args.feat if args.feat is not None else cfg.get("feat", "xyz"),
        num_class=args.num_class,
        dropout=args.dropout,
        graph_strategy=args.graph_strategy,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience if args.early_stop_patience is not None else cfg.get("early_stop_patience", 10),
        batch=args.batch,
        lr=args.lr,
        wd=args.wd,
        label_smoothing=args.label_smoothing,
        seed=args.seed if args.seed is not None else cfg.get("seed", 42),
        device=args.device,
        save_dir=args.save_dir if args.save_dir is not None else cfg.get("save_dir", "runs/indian_sign"),
        exp_name=args.exp_name if args.exp_name is not None else cfg.get("exp_name", "stgcn_indian_sign_v1"),
        offline_aug_factor=args.offline_aug_factor if args.offline_aug_factor is not None else cfg.get("offline_aug_factor", 0),
        offline_cache_dir=args.offline_cache_dir if args.offline_cache_dir is not None else cfg.get("offline_cache_dir"),
        build_cache=args.build_cache if args.build_cache is not None else cfg.get("build_cache", False),
    )
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # IO / logging
    save_root = Path(cfg.get("save_dir", "runs/indian_sign"))
    exp_name  = cfg.get("exp_name", "stgcn_indian_sign_v1")
    workdir   = save_root / exp_name
    workdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(workdir / "tf"))
    logs_dir = workdir / "logs"; logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "train_val_metrics.jsonl"

    # Dirs
    train_dir = cfg.get("train_dir")
    val_dir   = cfg.get("val_dir")

    if train_dir is None:
        raise RuntimeError("train_dir must be provided in config or via --train-dir (point to split folder e.g. train).")
    if val_dir is None:
        raise RuntimeError("val_dir must be provided in config or via --val-dir (point to split folder e.g. validation).")

    # Datasets & loaders
    max_T = int(cfg.get("max_T", 30))
    feat = cfg.get("feat", "xyz").lower()
    batch = int(cfg.get("batch", 32))
    offline_factor = int(cfg.get("offline_aug_factor", 0))
    cache_dir = cfg.get("offline_cache_dir", None)
    build_cache_flag = bool(cfg.get("build_cache", False))

    # Instantiate base datasets (they scan their root split dir)
    train_ds_base = IndianSignSequenceDataset(
        root_split_dir=str(train_dir),
        max_T=max_T,
        feat=feat,
        normalize=bool(cfg.get("normalize", True)),
        temporal_mode=cfg.get("temporal_mode", "interp"),
        aug=False,
        eval_mode=False,
        offline_aug_factor=0,   # base has no offline copies; we'll use dataset-level offline cache if enabled
    )
    val_ds = IndianSignSequenceDataset(
        root_split_dir=str(val_dir),
        max_T=max_T,
        feat=feat,
        normalize=bool(cfg.get("normalize", True)),
        temporal_mode=cfg.get("temporal_mode", "interp"),
        aug=False,
        eval_mode=True,
        offline_aug_factor=0,
    )

    # Determine number of classes from train_ds_base
    num_classes = len(train_ds_base.class_names)
    cfg["num_class"] = num_classes

    # Optionally enable dataset-level offline augmentation (the dataset supports offline_aug_factor and cache)
    if offline_factor > 0:
        if cache_dir is None:
            raise ValueError("Provide offline_cache_dir in config when offline_aug_factor > 0")
        # Re-create train dataset with offline aug enabled (it will build cache on first access if build_cache True)
        train_ds = IndianSignSequenceDataset(
            root_split_dir=str(train_dir),
            max_T=max_T,
            feat=feat,
            normalize=bool(cfg.get("normalize", True)),
            temporal_mode=cfg.get("temporal_mode", "interp"),
            aug=False,
            eval_mode=False,
            offline_aug_factor=offline_factor,
            offline_cache_dir=str(cache_dir),
            build_cache=build_cache_flag,
        )
    else:
        # use base
        train_ds = train_ds_base

    # DataLoaders
    num_workers = int(cfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate_sign,
                              worker_init_fn=_winit, generator=_g, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_sign,
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

    # Graph visualization (optional)
    try:
        example_batch = next(iter(train_loader))
        x_example = example_batch["data"].to(device)
        model_for_graph = model.module if hasattr(model, "module") else model
        print("[tb] Adding model graph to TensorBoard...")
        writer.add_graph(model_for_graph, x_example)
        writer.flush()
        print("[tb] Model graph added.")
        if make_dot is not None:
            print("[torchviz] Building PDF graph...")
            model_for_graph.eval()
            y_example = model_for_graph(x_example)
            if isinstance(y_example, (list, tuple)):
                y_out = y_example[0]
            else:
                y_out = y_example
            dot = make_dot(y_out, params=dict(model_for_graph.named_parameters()),
                           show_attrs=False, show_saved=False)
            graphs_dir = workdir / "graphs"; graphs_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = graphs_dir / f"{cfg.get('model','model_graph')}"
            dot.render(str(pdf_path), format="pdf", cleanup=True)
            print(f"[torchviz] PDF graph saved to: {pdf_path.with_suffix('.pdf')}")
        else:
            print("[torchviz] make_dot not available; skipped PDF export.")
    except Exception as e:
        print(f"[graph] Skipped graph export due to error: {e}")

    # Run header
    C = 3 if feat == "xyz" else 6
    T = int(cfg.get("max_T", max_T)); V = train_ds.V
    hdr_cfg = {
        "model": cfg.get("model","stgcn"),
        "feat": feat,
        "T": T, "V": V, "batch": int(batch),
        "epochs": int(cfg.get("epochs",100)),
        "optimizer": "AdamW",
        "lr": cfg.get("lr",1e-3), "wd": cfg.get("wd",1e-4),
        "label_smoothing": cfg.get("label_smoothing",0.0),
        "dropout": cfg.get("dropout",0.05),
        "aug_train": bool(cfg.get("aug", False)),
        "temporal_mode": cfg.get("temporal_mode","interp"),
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

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))

    epochs = int(cfg.get("epochs", 100))
    final_lr = 1e-6
    warmup_epochs = max(1, int(0.05 * epochs))
    sched_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    sched_cosine  = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=final_lr)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])

    # CFOT schedule
    cfot_start = int(cfg.get("cfot_start_epoch", 5))
    cfot_warm  = int(cfg.get("cfot_warmup_epochs", 20))
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
