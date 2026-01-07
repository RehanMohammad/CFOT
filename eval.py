#!/usr/bin/env python3
# Evaluation on SHREC'24 with ST-GCN/CFOT (directory-based)
# - Uses a directory of class subfolders as the test set
# - No augmentations at eval
# - Exports confusion matrix (csv/png), per-class accuracy, JSONL summary, and optional raw preds

from __future__ import annotations
import json, time, os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Optional sklearn
try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception:
    confusion_matrix = None
    classification_report = None

# Headless matplotlib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# Dataset
from dataset.shrec24 import SHREC24Dataset, shrec24_collate

# ------------------------------ small utils ------------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def override(cfg: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    for k, v in kwargs.items():
        if v is not None:
            cfg[k] = v
    return cfg

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_deterministic(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_count(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.2f}K"
    return str(n)

def opt_int(v, default: Optional[int]=None):
    if v is None: return default
    return int(v)

def opt_float(v, default: Optional[float]=None):
    if v is None: return default
    return float(v)

def opt_bool(v, default: Optional[bool]=None):
    if v is None: return default
    return bool(v)

def opt_list(v, default=None):
    if v is None: return default
    return v

def topk_from_logits_np(logits: np.ndarray, labels: np.ndarray, ks: Tuple[int, ...]=(1,5)) -> Dict[int,float]:
    t = torch.from_numpy(logits)
    y = torch.from_numpy(labels)
    C = t.size(1)
    out = {}
    _, topk = t.topk(min(max(ks), C), dim=1, largest=True, sorted=True)
    for k in ks:
        k = min(k, C)
        out[k] = (topk[:, :k] == y.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
    return out

def plot_confusion(cm: np.ndarray, out_png: Path, class_names: Optional[List[str]] = None, normalize: bool = True):
    if normalize:
        with np.errstate(all="ignore"):
            cmn = cm / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
    else:
        cmn = cm.astype(float)

    C = cm.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    vmin, vmax = (0.0, 1.0) if normalize else (0.0, cm.max() if cm.max() > 0 else 1.0)
    im = ax.imshow(cmn, cmap="Blues", vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized" if normalize else "Count", rotation=90)

    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if class_names is not None:
        ax.set_xticks(np.arange(C), labels=class_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(C), labels=class_names)
    else:
        ax.set_xticks(np.arange(C)); ax.set_yticks(np.arange(C))
    ax.set_aspect("equal"); ax.set_xlim(-0.5, C-0.5); ax.set_ylim(C-0.5, -0.5)
    for g in np.arange(-0.5, C, 1.0):
        ax.axhline(g, color="white", linewidth=0.5)
        ax.axvline(g, color="white", linewidth=0.5)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    def txt_color(val):
        r,g,b,_ = mpl.cm.Blues(norm(val))
        def lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
        L = 0.2126*lin(r)+0.7152*lin(g)+0.0722*lin(b)
        return "white" if L < 0.35 else "black"

    show = cmn if normalize else cm
    for i in range(C):
        for j in range(C):
            n = int(cm[i, j])
            val = show[i, j]
            if normalize and i != j and (val < 0.005 or n == 0):
                continue
            s = f"{100*val:.1f}%\n({n})" if normalize and n>0 else (f"{100*val:.1f}%" if normalize else str(n))
            ax.text(j, i, s, ha="center", va="center", fontsize=8,
                    fontweight=("bold" if i==j else "normal"), color=txt_color(val))
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("Confusion Matrix")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    try: fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    except Exception: pass
    plt.close(fig)

def maybe_restore_cfot_runtime(model: nn.Module, ckpt: Dict[str, Any]):
    if hasattr(model, "set_cfot_strength"):
        if "cfot_strength_frac" in ckpt:
            frac = float(ckpt["cfot_strength_frac"])
            model.set_cfot_strength(frac)
            print(f"[eval] CFOT strength fraction = {frac:.3f} (from checkpoint).")
        else:
            # Default to 1.0 if CFOT module exists and no runtime stored
            if getattr(model, "cfot_module", None) is not None:
                model.set_cfot_strength(1.0)
                print("[eval] CFOT strength fraction defaulted to 1.000.")

# ------------------------------ eval loop ------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, labels_all = [], []
    fwd_s, bsz = [], []

    wall0 = time.perf_counter()
    for batch in loader:
        x = batch["data"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()

        logits_all.append(out.detach().cpu())
        labels_all.append(y.detach().cpu())
        fwd_s.append(t1 - t0)
        bsz.append(y.size(0))
    wall = time.perf_counter() - wall0

    logits = torch.cat(logits_all, dim=0)  # [N,C]
    labels = torch.cat(labels_all, dim=0)  # [N]
    preds  = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    fwd_total = float(np.sum(fwd_s)) if fwd_s else 0.0
    N = int(labels.numel())
    timing = {
        "wall_total_s": wall,
        "forward_total_s": fwd_total,
        "avg_ms_per_sample_forward": (1000.0 * fwd_total / max(1, N)),
        "avg_ms_per_sample_wall": (1000.0 * wall / max(1, N)),
        "throughput_samples_per_s_forward": (N / fwd_total) if fwd_total > 0 else 0.0,
        "throughput_samples_per_s_wall": (N / wall) if wall > 0 else 0.0,
        "per_batch_forward_s": fwd_s,
        "per_batch_size": bsz,
    }
    return acc, preds.numpy(), labels.numpy(), logits.numpy(), timing

# ------------------------------ model builder ------------------------------

def _safe_build(cls, base_kwargs: Dict[str, Any], maybe_kwargs: Dict[str, Any]):
    try:
        return cls(**base_kwargs, **maybe_kwargs)
    except TypeError:
        import inspect
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys())
        cleaned = {k: v for k, v in maybe_kwargs.items() if k in allowed}
        try:
            return cls(**base_kwargs, **cleaned)
        except TypeError:
            return cls(**base_kwargs)

def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model_name = (cfg.get("model") or "stgcn").lower()
    feat = cfg.get("feat", "xyz")
    in_ch = 3 if feat == "xyz" else 6
    V = 28

    graph_args = {
        "layout": "shrec24",
        "strategy": cfg.get("graph_strategy", "spatial"),
        "max_hop": int(cfg.get("max_hop", 1)),
    }

    if model_name in ("twostreammsg3d","two_stream_msg3d","twostream","msg3d_2s","2s_msg3d","msg3d_two_stream"):
        from models.MSG3D.msg3d_two_stream import TwoStreamMSG3D
        model = TwoStreamMSG3D(
            num_class=int(cfg.get("num_class", 7)),
            num_point=V, num_person=1,
            graph="utils.graph.Graph",
            graph_args={"layout":"shrec24","strategy":graph_args["strategy"],"max_hop":graph_args["max_hop"]},
            in_channels=in_ch,
            fusion=cfg.get("fusion","mean_logits"),
            enable_cfot=bool(cfg.get("enable_cfot", False)),
            cfot_hidden=int(cfg.get("cfot_hidden", 64)),
            cfot_topk=int(cfg.get("cfot_topk", 3)),
            cfot_tau=float(cfg.get("cfot_tau", 0.45)),
            cfot_iters=int(cfg.get("cfot_iters", 9)),
            cfot_beta=float(cfg.get("cfot_beta", 0.6)),
            cfot_deltas=cfg.get("cfot_deltas", [1,2,4,8]),
            drop_out=float(cfg.get("dropout", 0.1)),
        ).to(device)
        return model

    if model_name == "agcn":
        from models.agcn.agcn import Model as AGCN
        return AGCN(
            in_channels=in_ch,
            num_class=int(cfg.get("num_class", 7)),
            graph_args=graph_args,
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.15)),
        ).to(device)

    if model_name == "ctrgcn":
        from models.ctrgcn.ctrgcn import Model as CTRGCN
        return CTRGCN(
            in_channels=in_ch,
            num_class=int(cfg.get("num_class", 7)),
            graph_args=graph_args,
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.10)),
        ).to(device)

    # default ST-GCN
    from models.stgcn.stgcn import Model as STGCN
    base = dict(
        in_channels=in_ch,
        num_class=int(cfg.get("num_class", 7)),
        graph_args=graph_args,
        edge_importance_weighting=True,
        dropout=float(cfg.get("dropout", 0.05)),
    )
    maybe = dict(
        enable_cfot=bool(cfg.get("enable_cfot", False)),
        cfot_type=cfg.get("cfot_type", "adaptive"),
        cfot_deltas=cfg.get("cfot_deltas", [1,2,4,8]),
        cfot_hidden=int(cfg.get("cfot_hidden", 64)),
        cfot_iters=int(cfg.get("cfot_iters", 9)),
        cfot_tau=float(cfg.get("cfot_tau", 0.45)),
        cfot_topk=int(cfg.get("cfot_topk", 6)),
        cfot_beta=float(cfg.get("cfot_beta", 0.6)),
        cfot_inject=cfg.get("cfot_inject", "pre"),
        temporal_kernel_size=int(cfg.get("temporal_kernel_size", 9)),

        # adaptive sparsity
        cfot_sparsify=cfg.get("cfot_sparsify", "adaptive"),
        cfot_keep_mass=float(cfg.get("cfot_keep_mass", 0.4)),
        cfot_min_k=int(cfg.get("cfot_min_k", 2)),
        cfot_max_k=int(cfg.get("cfot_max_k", V) or V),

        # learned affinity extras
        affinity=cfg.get("cfot_affinity", cfg.get("affinity", "learned")),
        metric_rank=cfg.get("cfot_metric_rank", cfg.get("metric_rank", None)),
        vel_in_learned=bool(cfg.get("cfot_vel_in_learned", cfg.get("vel_in_learned", False))),
        pos_w=cfg.get("pos_w", 1.0),
        vel_w=cfg.get("vel_w", 0.2),
    )
    return _safe_build(STGCN, base, maybe).to(device)

# ------------------------------ main ------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)

    # data
    ap.add_argument("--test-data-dir", type=str, required=True,
                    help="Directory with class subfolders for SHREC'24 test set.")
    ap.add_argument("--max-T", type=int, default=None)
    ap.add_argument("--normalize", dest="normalize", action="store_true")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=None)
    ap.add_argument("--feat", type=str, default=None, choices=["xyz","xyz+vel"])
    ap.add_argument("--temporal-mode", type=str, default=None, choices=["crop_repeat","interp"])

    # model and runtime
    ap.add_argument("--model", type=str, default=None,
                    choices=["stgcn","msg3d","twostreammsg3d","msg3d_two_stream","2s_msg3d","agcn","ctrgcn"])
    ap.add_argument("--num-class", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--graph-strategy", type=str, default=None)
    ap.add_argument("--max-hop", type=int, default=None)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--dump-preds", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # CFOT optional
    ap.add_argument("--enable-cfot", dest="enable_cfot", action="store_true")
    ap.add_argument("--disable-cfot", dest="enable_cfot", action="store_false")
    ap.set_defaults(enable_cfot=None)
    ap.add_argument("--cfot-inject", type=str, choices=["pre","after1"], default=None)
    ap.add_argument("--cfot-sparsify", type=str, choices=["topk","adaptive"], default=None)
    ap.add_argument("--cfot-keep-mass", type=float, default=None)
    ap.add_argument("--cfot-min-k", type=int, default=None)
    ap.add_argument("--cfot-max-k", type=int, default=None)
    ap.add_argument("--cfot-affinity", type=str, choices=["learned","euclid"], default=None)
    ap.add_argument("--cfot-metric-rank", type=int, default=None)
    ap.add_argument("--cfot-vel-in-learned", dest="cfot_vel_in_learned", action="store_true")
    ap.add_argument("--pos-w", type=float, default=None)
    ap.add_argument("--vel-w", type=float, default=None)

    args = ap.parse_args()
    set_deterministic(args.seed)

    # config + overrides
    cfg = load_config(args.config)
    cfg = override(cfg,
        # data
        normalize=(cfg.get("normalize", True) if args.normalize is None else bool(args.normalize)),
        feat=args.feat if args.feat is not None else cfg.get("feat", "xyz"),
        temporal_mode=args.temporal_mode if args.temporal_mode is not None else cfg.get("temporal_mode", "crop_repeat"),
        max_T=args.max_T if args.max_T is not None else cfg.get("max_T", 600),

        # model/runtime
        model=args.model if args.model is not None else cfg.get("model", "stgcn"),
        num_class=args.num_class if args.num_class is not None else cfg.get("num_class", 7),
        dropout=args.dropout if args.dropout is not None else cfg.get("dropout", 0.05),
        graph_strategy=args.graph_strategy if args.graph_strategy is not None else cfg.get("graph_strategy", "spatial"),
        max_hop=args.max_hop if args.max_hop is not None else cfg.get("max_hop", 1),
        batch=args.batch,
        device=args.device if args.device is not None else cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),

        # CFOT
        enable_cfot=(cfg.get("enable_cfot", False) if args.enable_cfot is None else bool(args.enable_cfot)),
        cfot_inject=args.cfot_inject if args.cfot_inject is not None else cfg.get("cfot_inject", "pre"),
        cfot_sparsify=args.cfot_sparsify if args.cfot_sparsify is not None else cfg.get("cfot_sparsify", "adaptive"),
        cfot_keep_mass=args.cfot_keep_mass if args.cfot_keep_mass is not None else cfg.get("cfot_keep_mass", 0.4),
        cfot_min_k=args.cfot_min_k if args.cfot_min_k is not None else cfg.get("cfot_min_k", 2),
        cfot_max_k=args.cfot_max_k if args.cfot_max_k is not None else cfg.get("cfot_max_k", 28),
        cfot_affinity=args.cfot_affinity if args.cfot_affinity is not None else cfg.get("cfot_affinity", cfg.get("affinity", "learned")),
        cfot_metric_rank=args.cfot_metric_rank if args.cfot_metric_rank is not None else cfg.get("cfot_metric_rank", cfg.get("metric_rank", None)),
        cfot_vel_in_learned=bool(args.cfot_vel_in_learned) if args.cfot_vel_in_learned is not None else bool(cfg.get("cfot_vel_in_learned", cfg.get("vel_in_learned", False))),
        pos_w=args.pos_w if args.pos_w is not None else cfg.get("pos_w", 1.0),
        vel_w=args.vel_w if args.vel_w is not None else cfg.get("vel_w", 0.2),
    )

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # resolve checkpoint and output dir
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    workdir = ckpt_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else (workdir / "eval_shrec24")
    ensure_dir(out_dir)

    raw = torch.load(ckpt_path, map_location="cpu")
    ckpt: Dict[str, Any] = raw if isinstance(raw, dict) and "model" in raw else {"model": raw}

    # dataset (directory of class subfolders). No aug at test.
    test_ds = SHREC24Dataset(
        data_dir=args.test_data_dir,
        max_T=int(cfg.get("max_T", 600)),
        normalize=bool(cfg.get("normalize", True)),
        feat=cfg.get("feat", "xyz"),
        temporal_mode=cfg.get("temporal_mode", "crop_repeat"),
        aug=False,
        eval_mode=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg.get("batch", 16)), shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=shrec24_collate
    )

    # class names from dataset if provided
    class_names = getattr(test_ds, "class_names", None)
    if not class_names:
        class_names = [str(i) for i in range(int(cfg.get("num_class", 7)))]

    # model
    model = build_model(cfg, device)
    print(f"[eval] Model: {type(model).__name__} | params: {human_count(count_params(model))}")
    print(f"[eval] Data: feat={cfg.get('feat')} norm={cfg.get('normalize')} "
          f"T_mode={cfg.get('temporal_mode')} max_T={cfg.get('max_T')} "
          f"| test_dir={args.test_data_dir}")

    # load weights
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception as e:
        print(f"[eval][WARN] strict load failed ({e}). Retrying strict=False...")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if unexpected:
            print(f"[eval][WARN] Unexpected keys: {list(unexpected)}")
        if missing:
            preview = list(missing)[:10]
            print(f"[eval][WARN] Missing keys (first 10): {preview}{' ...' if len(missing)>10 else ''}")

    # CFOT runtime
    maybe_restore_cfot_runtime(model, ckpt)

    # run eval
    acc, preds, labels, logits, timing = evaluate(model, test_loader, device)
    N = len(labels)
    tks = topk_from_logits_np(logits, labels, ks=(1,5)) if logits.shape[1] >= 5 else {1: acc*100.0}

    print(f"Test accuracy: {acc*100:.2f}%  (N={N})")
    if 5 in tks:
        print(f"Top-1: {tks[1]:.2f}%  | Top-5: {tks[5]:.2f}%")
    print(f"[timing] forward_total={timing['forward_total_s']:.3f}s | wall_total={timing['wall_total_s']:.3f}s | "
          f"avg_forward={timing['avg_ms_per_sample_forward']:.3f} ms/sample | "
          f"avg_wall={timing['avg_ms_per_sample_wall']:.3f} ms/sample | "
          f"throughput_forward={timing['throughput_samples_per_s_forward']:.1f} samp/s | "
          f"throughput_wall={timing['throughput_samples_per_s_wall']:.1f} samp/s")

    # confusion matrix
    C = len(class_names)
    if confusion_matrix is None:
        cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(labels, preds):
            if 0 <= t < C and 0 <= p < C:
                cm[t, p] += 1
    else:
        cm = confusion_matrix(labels, preds, labels=list(range(C)))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with np.errstate(all="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    np.savetxt(out_dir / "confusion_matrix_norm.csv", np.nan_to_num(cmn), fmt="%.6f", delimiter=",")
    plot_confusion(cm, out_dir / "confusion_matrix.png", class_names=class_names, normalize=True)

    # per-class recall
    row_sum = cm.sum(axis=1, keepdims=False).clip(min=1)
    per_class_recall = (cm.diagonal() / row_sum).astype(np.float64)
    np.savetxt(out_dir / "per_class_accuracy.csv", per_class_recall, fmt="%.6f", delimiter=",")

    # classification report
    if classification_report is not None:
        report = classification_report(labels, preds, labels=list(range(C)),
                                       target_names=class_names, digits=4)
        with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

    # optional raw preds
    if args.dump_preds:
        import pandas as pd
        probs = torch.from_numpy(logits).softmax(dim=1).numpy()
        df = pd.DataFrame({
            "label": labels,
            "pred": preds,
            "conf": probs[np.arange(len(preds)), preds],
        })
        df.to_csv(out_dir / "predictions.csv", index=False)
        print(f"Saved raw predictions: {out_dir / 'predictions.csv'}")

    # per-batch timing
    per_batch = np.column_stack([
        np.arange(len(timing["per_batch_forward_s"]), dtype=np.int64),
        np.array(timing["per_batch_size"], dtype=np.int64),
        np.array(timing["per_batch_forward_s"], dtype=np.float64),
        1000.0 * np.array(timing["per_batch_forward_s"], dtype=np.float64) / np.maximum(1, np.array(timing["per_batch_size"], dtype=np.float64)),
    ])
    header = "batch_idx,batch_size,forward_s,per_sample_ms"
    np.savetxt(out_dir / "per_batch_timing.csv", per_batch, delimiter=",", fmt="%s", header=header)

    # summary JSONL
    summary = {
        "split": "test",
        "N": int(N),
        "acc": float(acc),
        "top1": float(tks.get(1, acc*100.0)),
        "top5": float(tks.get(5, 0.0)),
        "forward_total_s": float(timing["forward_total_s"]),
        "wall_total_s": float(timing["wall_total_s"]),
        "avg_ms_per_sample_forward": float(timing["avg_ms_per_sample_forward"]),
        "avg_ms_per_sample_wall": float(timing["avg_ms_per_sample_wall"]),
        "throughput_samples_per_s_forward": float(timing["throughput_samples_per_s_forward"]),
        "throughput_samples_per_s_wall": float(timing["throughput_samples_per_s_wall"]),
    }
    with open(out_dir / "eval_metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"Saved eval artifacts to: {out_dir}")

if __name__ == "__main__":
    main()
