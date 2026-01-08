#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFOT-aware evaluation for ST-GCN/MSG3D family on Briareo.

Key features
------------
• Auto-enable CFOT when the checkpoint contains cfot_module weights.
• --enable-cfot / --disable-cfot to force CFOT on/off at eval time.
• Restores cfot_strength_frac from the checkpoint (so test == train behavior).
• Optional strict mode to error if CFOT is requested but not present.
• Exports confusion matrix (CSV/PNG), per-class recall CSV, optional predictions CSV.
• Prints timing (throughput, avg ms/sample) and Top-1/Top-5.

Example
-------
python eval_briareo.py  --config configs/briareo_stgcn.yaml   --ann-test "D:\\Dataset\\Briareo_landmarks_splits\\splits\\test\\depth_test.npz" --ckpt runs\\briareo_stgcn\\stgcn_briareo_xyz_seed_42_adaptive_adaptive_84\\best.pt --dump-preds
"""

from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# sklearn guarded import
try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception:
    confusion_matrix = None
    classification_report = None

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from dataset.Briareo import BriareoDataset, build_briareo_label_map, briareo_collate


# ------------------------------
# Small utils
# ------------------------------
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

def ckpt_requires_cfot(state: Dict[str, Any]) -> bool:
    keys = list(state.keys())
    return any(k.startswith("cfot_module.") or ".cfot_module." in k or k.startswith("backbone.cfot_module.")
               for k in keys)

def maybe_restore_cfot_runtime(model: nn.Module, ckpt: Dict[str, Any]):
    if not hasattr(model, "set_cfot_strength"):
        return
    if "cfot_strength_frac" in ckpt:
        frac = float(ckpt["cfot_strength_frac"])
        note = "from checkpoint (cfot_strength_frac)"
    elif "cfot_runtime" in ckpt:
        beta_max = float(getattr(model, "cfot_beta", 1.0) or 1.0)
        runtime = float(ckpt["cfot_runtime"])
        frac = 0.0 if beta_max == 0 else max(0.0, min(1.0, runtime / beta_max))
        note = "from checkpoint (cfot_runtime→fraction)"
    else:
        if getattr(model, "cfot_module", None) is None:
            return
        frac = 1.0
        note = "defaulted to 1.0 (no CFOT runtime stored)"
    model.set_cfot_strength(frac)
    print(f"[eval] CFOT strength fraction = {frac:.3f} ({note}).")

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
            if normalize and i != j and (val < 0.005 or n == 0):  # declutter
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


# ------------------------------
# Eval loop (timed)
# ------------------------------
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


# ------------------------------
# Main
# ------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--ann-test", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=None)

    # overrides / runtime
    ap.add_argument("--num-class", type=int, default=None)
    ap.add_argument("--in-ch", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--graph-strategy", type=str, default=None)
    ap.add_argument("--max-T", type=int, default=None)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--feat", type=str, default=None, choices=["xyz", "xyz+vel"])
    ap.add_argument("--temporal-mode", type=str, default=None, choices=["crop_repeat", "interp"])

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--dump-preds", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfot-inject", type=str, choices=["pre","after1"], default=None)

    # CFOT toggles
    ap.add_argument("--enable-cfot",  dest="enable_cfot", action="store_true",
                    help="Force-enable CFOT in the backbone (if supported).")
    ap.add_argument("--disable-cfot", dest="enable_cfot", action="store_false",
                    help="Disable CFOT even if config enables it.")
    ap.set_defaults(enable_cfot=None)
    ap.add_argument("--strict-cfot", action="store_true",
                    help="Error if CFOT is requested/needed but model lacks cfot_module.")
    
    ap.add_argument("--cfot-affinity", type=str, default=None, choices=["euclid","cosine","learned"])
    ap.add_argument("--cfot-euclid-scale", type=float, default=None)
    ap.add_argument("--cfot-cosine-eps", type=float, default=None)
    ap.add_argument("--cfot-pos-weight", type=float, default=None)
    ap.add_argument("--cfot-vel-weight", type=float, default=None)

    args = ap.parse_args()
    set_deterministic(args.seed)

    # Load cfg & apply overrides
    cfg = load_config(args.config)
    cfg = override(
        cfg,
        data_dir=args.data_dir,
        num_class=args.num_class,
        in_ch=args.in_ch,
        dropout=args.dropout,
        graph_strategy=args.graph_strategy,
        max_T=args.max_T,
        normalize=args.normalize if args.normalize else cfg.get("normalize", False),
        feat=args.feat if args.feat is not None else cfg.get("feat", "xyz"),
        temporal_mode=args.temporal_mode if args.temporal_mode is not None else cfg.get("temporal_mode", "interp"),
        batch=args.batch,
        device=args.device,
        cfot_inject=args.cfot_inject,
        cfot_affinity=args.cfot_affinity if args.cfot_affinity is not None else cfg.get("cfot_affinity","euclid"),
        cfot_euclid_scale=args.cfot_euclid_scale if args.cfot_euclid_scale is not None else cfg.get("cfot_euclid_scale",1.0),
        cfot_cosine_eps=args.cfot_cosine_eps if args.cfot_cosine_eps is not None else cfg.get("cfot_cosine_eps",1e-6),
        cfot_pos_weight=args.cfot_pos_weight if args.cfot_pos_weight is not None else cfg.get("cfot_pos_weight",1.0),
        cfot_vel_weight=args.cfot_vel_weight if args.cfot_vel_weight is not None else cfg.get("cfot_vel_weight",0.2),
    )

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Resolve checkpoint & output dir
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        workdir = ckpt_path.parent
    else:
        workdir = Path(cfg.get("save_dir", "runs/Briareo_stgcn")) / cfg.get("exp_name", "exp")
        ckpt_path = workdir / "best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else (workdir / "eval")
    ensure_dir(out_dir)

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    sd: Dict[str, Any] = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    # Decide CFOT usage: CLI > config > checkpoint auto-detect
    if args.enable_cfot is None:
        want_cfot = bool(cfg.get("enable_cfot", False))
        if not want_cfot and ckpt_requires_cfot(sd):
            want_cfot = True
            print("[eval][CFOT] Auto-enabled CFOT because checkpoint contains cfot_module weights.")
    else:
        want_cfot = bool(args.enable_cfot)


    label_map = build_briareo_label_map(cfg["ann_train"], cfg.get("data_dir"))

    # Dataset (no aug at test)
    test_ds = BriareoDataset(
        ann_file=args.ann_test,
        data_dir=cfg.get("data_dir"),
        max_T=int(cfg.get("max_T", 180)),
        normalize=bool(cfg.get("normalize", False)),
        feat=cfg.get("feat", "xyz"),
        temporal_mode=cfg.get("temporal_mode", "interp"),
        aug=False,
        eval_mode=True,
        label_map=label_map
    )

    num_classes = test_ds.num_classes
    cfg["num_class"] = num_classes

    test_loader = DataLoader(
        test_ds, batch_size=int(cfg.get("batch", 64)),
        shuffle=False, num_workers=4, pin_memory=True, collate_fn=briareo_collate
    )

    # Build the SAME backbone as train (pass CFOT kwargs too)
    model_name = cfg.get("model", "stgcn").lower()
    feat = cfg.get("feat", "xyz")
    in_ch = int(cfg.get("in_ch", (3 if feat == "xyz" else 6)))

    graph_args = {
        "layout": "briareo",
        "strategy": cfg.get("graph_strategy", "spatial"),
        "max_hop": int(cfg.get("max_hop", 1)),
    }

    if model_name in ("twostreammsg3d","two_stream_msg3d","twostream","msg3d_2s","2s_msg3d","msg3d_two_stream"):
        from models.MSG3D.msg3d_two_stream import TwoStreamMSG3D
        model = TwoStreamMSG3D(
            num_class=int(cfg.get("num_class", 13)),
            num_point=22, num_person=1,
            graph="utils.graph.Graph",
            graph_args={"layout":"briareo","strategy":cfg.get("graph_strategy","spatial"),"max_hop": int(cfg.get("max_hop",2))},
            in_channels=in_ch,
            fusion=cfg.get("fusion","mean_logits"),
            enable_cfot=want_cfot,
            cfot_hidden=int(cfg.get("cfot_hidden", 64)),
            cfot_topk=int(cfg.get("cfot_topk", 3)),
            cfot_tau=float(cfg.get("cfot_tau", 0.7)),
            cfot_iters=int(cfg.get("cfot_iters", 10)),
            cfot_beta=float(cfg.get("cfot_beta", 1.0)),
            cfot_deltas=cfg.get("cfot_deltas", [1,2]),
            drop_out=float(cfg.get("dropout", 0.1)),
        ).to(device)
    elif model_name == "msg3d":
        from models.MSG3D.msg3d import Model as MSG3D
        model = MSG3D(
            num_class=int(cfg.get("num_class", 13)),
            num_point=22, num_person=1,
            graph="utils.graph.Graph",
            graph_args={"layout":"briareo","strategy":cfg.get("graph_strategy","spatial"),"max_hop": int(cfg.get("max_hop",2))},
            in_channels=in_ch,
            drop_out=float(cfg.get("dropout", 0.1)),
            enable_cfot=want_cfot,
            cfot_hidden=int(cfg.get("cfot_hidden", 64)),
            cfot_topk=int(cfg.get("cfot_topk", 3)),
            cfot_tau=float(cfg.get("cfot_tau", 0.7)),
            cfot_iters=int(cfg.get("cfot_iters", 10)),
            cfot_beta=float(cfg.get("cfot_beta", 1.0)),
            cfot_deltas=cfg.get("cfot_deltas", [1,2]),
        ).to(device)
    elif model_name == "agcn":
        from models.agcn.agcn import Model as AGCN
        model = AGCN(
            in_channels=in_ch,
            num_class=int(cfg.get("num_class", 13)),
            graph_args=graph_args,
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.15)),
        ).to(device)
    elif model_name == "ctrgcn":
        from models.ctrgcn.ctrgcn import Model as CTRGCN
        model = CTRGCN(
            in_channels=in_ch,
            num_class=int(cfg.get("num_class", 13)),
            graph_args=graph_args,
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.10)),
        ).to(device)
    elif model_name == "stgcn":
        from models.stgcn.stgcn import Model as STGCN
        model = STGCN(
            in_channels=in_ch,
            num_class=int(cfg.get("num_class", 14)),
            graph_args=graph_args,
            edge_importance_weighting=True,
            dropout=float(cfg.get("dropout", 0.15)),

            # CFOT knobs (match train)
            enable_cfot=want_cfot,
            cfot_type=cfg.get("cfot_type", "adaptive"),
            cfot_deltas=cfg.get("cfot_deltas", [1, 2]),
            cfot_hidden=int(cfg.get("cfot_hidden", 64)),
            cfot_iters=int(cfg.get("cfot_iters", 10)),
            cfot_tau=float(cfg.get("cfot_tau", 0.7)),
            cfot_topk=int(cfg.get("cfot_topk", 3)),
            cfot_beta=float(cfg.get("cfot_beta", 1.0)),
            cfot_inject=cfg.get("cfot_inject", "pre"),
            temporal_kernel_size=int(cfg.get("temporal_kernel_size", 9)),

            # NEW sparsify knobs
            cfot_sparsify=cfg.get("cfot_sparsify", "topk"),
            cfot_keep_mass=float(cfg.get("cfot_keep_mass", 0.9)),
            cfot_min_k=int(cfg.get("cfot_min_k", 2)),
            cfot_max_k=int(cfg.get("cfot_max_k", None) or 21),
            cfot_affinity=cfg.get("cfot_affinity","euclid"),
            cfot_euclid_scale=float(cfg.get("cfot_euclid_scale",1.0)),
            cfot_cosine_eps=float(cfg.get("cfot_cosine_eps",1e-6)),
            cfot_pos_weight=float(cfg.get("cfot_pos_weight",1.0)),
            cfot_vel_weight=float(cfg.get("cfot_vel_weight",0.2)),
        ).to(device)
    else:
        raise SystemExit(f"Unknown model '{model_name}'.")

    requested = bool(want_cfot)
    present   = hasattr(model, "cfot_module") and (getattr(model, "cfot_module", None) is not None)
    if requested and not present:
        msg = ("[eval][CFOT][WARN] enable_cfot=True but the model has no cfot_module. "
               "Check the backbone's CFOT import/kwargs.")
        if args.strict_cfot: raise SystemExit(msg)
        print(msg)

    print(f"[eval] Model: {type(model).__name__} | params: {human_count(count_params(model))}")
    print(f"[eval] Data: feat={cfg.get('feat')}  norm={cfg.get('normalize')}  "
          f"T_mode={cfg.get('temporal_mode')}  max_T={cfg.get('max_T')}")

    # Load weights (strict; print helpful error for CFOT mismatch)
    ckpt = raw if isinstance(raw, dict) and "model" in raw else {"model": raw}
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception as e:
        needs_cfot = ckpt_requires_cfot(ckpt["model"])
        if needs_cfot and not present:
            raise SystemExit("[eval][CFOT] Checkpoint has CFOT weights but model lacks CFOT.\n"
                             "  → Re-run with --enable-cfot (and ensure your backbone attaches CFOT).")
        raise
    print(f"Loaded checkpoint: {ckpt_path} (epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')})")

    # Restore CFOT runtime (no tuning at test)
    maybe_restore_cfot_runtime(model, ckpt)

    # ---- Evaluate ----
    acc, preds, labels, logits, timing = evaluate(model, test_loader, device)
    N = len(labels)
    print(f"Test accuracy: {acc * 100:.2f}%  (N={N})")
    tks = topk_from_logits_np(logits, labels, ks=(1,5)) if logits.shape[1] >= 5 else {1: acc*100.0}
    if 5 in tks:
        print(f"Top-1: {tks[1]:.2f}%  | Top-5: {tks[5]:.2f}%")
    print(f"[timing] forward_total={timing['forward_total_s']:.3f}s | wall_total={timing['wall_total_s']:.3f}s | "
          f"avg_forward={timing['avg_ms_per_sample_forward']:.3f} ms/sample | "
          f"avg_wall={timing['avg_ms_per_sample_wall']:.3f} ms/sample | "
          f"throughput_forward={timing['throughput_samples_per_s_forward']:.1f} samp/s | "
          f"throughput_wall={timing['throughput_samples_per_s_wall']:.1f} samp/s")

    # ---- Artifacts ----
    class_names = [str(i) for i in range(int(cfg.get("num_class", 13)))]

    # Confusion matrix
    if confusion_matrix is None:
        cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for t, p in zip(labels, preds):
            if 0 <= t < len(class_names) and 0 <= p < len(class_names):
                cm[t, p] += 1
    else:
        cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with np.errstate(all="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    np.savetxt(out_dir / "confusion_matrix_norm.csv", np.nan_to_num(cmn), fmt="%.6f", delimiter=",")
    plot_confusion(cm, out_dir / "confusion_matrix.png", class_names=class_names, normalize=True)

    # Per-class recall
    row_sum = cm.sum(axis=1, keepdims=False).clip(min=1)
    per_class_recall = (cm.diagonal() / row_sum).astype(np.float64)
    np.savetxt(out_dir / "per_class_accuracy.csv", per_class_recall, fmt="%.6f", delimiter=",")

    # Classification report
    if classification_report is not None:
        report = classification_report(labels, preds, labels=list(range(len(class_names))),
                                       target_names=class_names, digits=4)
        with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

    # Optional raw predictions
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

    # Per-batch timing CSV
    per_batch = np.column_stack([
        np.arange(len(timing["per_batch_forward_s"]), dtype=np.int64),
        np.array(timing["per_batch_size"], dtype=np.int64),
        np.array(timing["per_batch_forward_s"], dtype=np.float64),
        1000.0 * np.array(timing["per_batch_forward_s"], dtype=np.float64) / np.maximum(1, np.array(timing["per_batch_size"], dtype=np.float64)),
    ])
    header = "batch_idx,batch_size,forward_s,per_sample_ms"
    np.savetxt(out_dir / "per_batch_timing.csv", per_batch, delimiter=",", fmt="%s", header=header)

    # Summary row
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
