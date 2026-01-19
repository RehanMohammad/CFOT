#!/usr/bin/env python3
# Evaluation on ISL with ST-GCN/CFOT
# - Loads YAML + CLI overrides
# - Optionally forces CFOT OFF at runtime without changing architecture (--disable-cfot)
# - Backfills missing CFOT knobs from checkpoint's saved cfg
# - Ensures eval-time CFOT affinity matches training unless explicitly overridden
# - Dumps predictions if requested


# python eval_isl.py --config configs/ISL_stgcn.yaml --ckpt runs/indian_sign/stgcn_indian_sign_v1/best.pth


from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project modules
from dataset.isl import (
    IndianSignSequenceDataset,
    collate_sign,
)

# Reuse helpers from training
from scripts.train_isl import (
    load_config,
    override,
    set_seed,
    build_model,
    accuracy_at_k,
)

# ------------------------------
# Local helpers
# ------------------------------

def force_reproducible(seed: int = 42):
    import os, torch, numpy as np, random
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
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

# in main(), as the very first call:
force_reproducible(42)


g = torch.Generator()
g.manual_seed(42)

def _winit(worker_id):
    wseed = 42 + worker_id
    np.random.seed(wseed)
    random.seed(wseed)



def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _write_csv(path: Path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header:
            f.write(header + "\n")
        for r in rows:
            if isinstance(r, (list, tuple)):
                f.write(",".join(str(v) for v in r) + "\n")
            else:
                f.write(str(r) + "\n")

def _merge_ckpt_cfg(cfg: Dict[str, Any], ckpt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Only fill missing keys from checkpoint config
    keys = [
    "cfot_affinity", "cfot_euclid_scale", "cfot_cosine_eps",
    "cfot_pos_weight", "cfot_vel_weight",
    "cfot_type", "cfot_deltas", "cfot_hidden",
    "cfot_iters", "cfot_tau", "cfot_topk", "cfot_inject",
    "enable_cfot",
    "feat", "max_T", "normalize", "temporal_mode",
    "num_class", "dropout", "graph_strategy", "max_hop",
    "save_dir", "exp_name"
    ]

    out = dict(cfg)
    for k in keys:
        if (k not in out) or (out[k] is None):
            if k in ckpt_cfg:
                out[k] = ckpt_cfg[k]
    return out

def hard_disable_cfot(model: nn.Module):
    """
    Force CFOT off at runtime without removing the module.
    Safe for models built with CFOT so state_dict still loads.
    """
    # Top-level switches/values commonly used in CFOT-enabled ST-GCNs
    if hasattr(model, "enable_cfot"):
        try: model.enable_cfot = False
        except Exception: pass
    if hasattr(model, "set_cfot_strength"):
        try: model.set_cfot_strength(0.0)
        except Exception: pass
    if hasattr(model, "cfot_strength_frac"):
        try: model.cfot_strength_frac = 0.0
        except Exception: pass
    if hasattr(model, "cfot_beta"):
        try: model.cfot_beta = 0.0
        except Exception: pass
    if hasattr(model, "cfot_affinity"):
        try: model.cfot_affinity = "euclid"
        except Exception: pass

    # Nested module
    m = getattr(model, "cfot_module", None)
    if m is not None:
        for k in ("enabled", "enable", "use_cfot"):
            if hasattr(m, k):
                try: setattr(m, k, False)
                except Exception: pass
        for k in ("strength", "strength_frac", "lambda_cfot", "beta", "gate_tau"):
            if hasattr(m, k):
                try: setattr(m, k, 0.0)
                except Exception: pass
        # optional runtime kill switch if implemented
        if hasattr(m, "disable"):
            try: m.disable(True)
            except Exception: pass
    # Optional model-level runtime kill switch
    if hasattr(model, "disable_cfot_runtime"):
        try: model.disable_cfot_runtime(True)
        except Exception: pass

# ------------------------------
# Eval
# ------------------------------

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device,
             dump_preds_path: Optional[Path] = None) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    top1_meter = 0.0
    top5_meter = 0.0
    losses: List[float] = []
    rows = []

    criterion = nn.CrossEntropyLoss()

    fwd_t0 = time.perf_counter()
    wall_t0 = fwd_t0
    n_samples = 0

    for batch in loader:
        x = batch["data"].to(device, non_blocking=True)    # expected [N,C,T,V,1] from collate
        y = batch["label"].to(device, non_blocking=True)

        t0 = time.perf_counter()
        logits = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = time.perf_counter() - t0  # per-batch forward time if needed

        loss = criterion(logits, y)
        bs = y.size(0)
        total += bs
        n_samples += bs
        losses.append(float(loss.item()))

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())

        accs = accuracy_at_k(logits, y, ks=(1, 5))
        top1_meter += accs[1] * bs / 100.0
        top5_meter += accs[5] * bs / 100.0

        if dump_preds_path is not None:
            probs = torch.softmax(logits, dim=1)
            top5p, top5i = probs.topk(min(5, probs.size(1)), dim=1)
            for i in range(bs):
                rows.append([
                    len(rows),
                    int(y[i].item()),
                    int(pred[i].item()),
                    *[int(top5i[i, j].item()) for j in range(top5i.size(1))],
                    *[float(top5p[i, j].item()) for j in range(top5p.size(1))],
                ])

    forward_total = time.perf_counter() - fwd_t0
    wall_total = time.perf_counter() - wall_t0

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    acc = 100.0 * (correct / max(1, total))
    top1 = 100.0 * (top1_meter / max(1, total))
    top5 = 100.0 * (top5_meter / max(1, total))

    if dump_preds_path is not None:
        header = "row,true,pred,top1,top2,top3,top4,top5,p1,p2,p3,p4,p5"
        _write_csv(dump_preds_path, rows, header=header)

    return {
        "loss": avg_loss,
        "acc": acc,
        "top1": top1,
        "top5": top5,
        "forward_total": forward_total,
        "wall_total": wall_total,
        "avg_forward_ms": 1000.0 * forward_total / max(1, n_samples),
        "avg_wall_ms": 1000.0 * wall_total / max(1, n_samples),
        "throughput_forward": n_samples / max(1e-9, forward_total),
        "throughput_wall": n_samples / max(1e-9, wall_total),
    }

# ------------------------------
# Main
# ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--feat", type=str, default=None, choices=["xyz", "xyz+vel"])
    p.add_argument("--max-T", type=int, default=None)
    p.add_argument("--normalize", dest="normalize", action="store_true")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.set_defaults(normalize=None)
    p.add_argument("--temporal-mode", type=str, default=None, choices=["crop_repeat", "interp"])
    p.add_argument("--seed", type=int, default=None)

    # CFOT knobs: we do NOT flip architecture via CLI; we only allow runtime disable.
    p.add_argument("--cfot-type", type=str, default=None, choices=["adaptive", "multi", "single"])
    p.add_argument("--cfot-deltas", type=int, nargs="+", default=None)
    p.add_argument("--cfot-hidden", type=int, default=None)
    p.add_argument("--cfot-iters", type=int, default=None)
    p.add_argument("--cfot-tau", type=float, default=None)
    p.add_argument("--cfot-topk", type=int, default=None)
    p.add_argument("--cfot-inject", type=str, default=None, choices=["replace_all", "pre", "after1", "replace_first"])
    p.add_argument("--cfot-beta", type=float, default=None)

    # Affinity ablation
    p.add_argument("--cfot-affinity", type=str, default=None, choices=["euclid", "cosine", "learned"])
    p.add_argument("--cfot-euclid-scale", type=float, default=None)
    p.add_argument("--cfot-cosine-eps", type=float, default=None)
    p.add_argument("--cfot-pos-weight", type=float, default=None)
    p.add_argument("--cfot-vel-weight", type=float, default=None)

    # Runtime kill switch
    p.add_argument("--disable-cfot", action="store_true", help="Force CFOT off at eval time without changing the model architecture.")

    # Outputs
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--dump-preds", action="store_true")

    args = p.parse_args()

    # Load config
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    if "root" not in data_cfg or "splits" not in data_cfg or "test" not in data_cfg["splits"]:
        raise KeyError("Config must contain data.root and data.splits.test for ISL evaluation.")

    cfg["test_dir"] = str(
        Path(data_cfg["root"]) / data_cfg["splits"]["test"]
    )


    # Basic overrides from CLI (do not overwrite enable_cfot here)
    cfg = override(cfg,
    batch=args.batch if args.batch is not None else cfg.get("batch", 64),
    device=args.device if args.device is not None else cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    feat=args.feat if args.feat is not None else cfg.get("feat", "xyz"),
    max_T=args.max_T if args.max_T is not None else cfg.get("max_T", 80),
    normalize=cfg.get("normalize", False) if args.normalize is None else bool(args.normalize),
    temporal_mode=args.temporal_mode if args.temporal_mode is not None else cfg.get("temporal_mode", "interp"),
    seed=args.seed if args.seed is not None else cfg.get("seed", 42),

    # CFOT knobs
    cfot_type=args.cfot_type if args.cfot_type is not None else cfg.get("cfot_type"),
    cfot_deltas=args.cfot_deltas if args.cfot_deltas is not None else cfg.get("cfot_deltas"),
    cfot_hidden=args.cfot_hidden if args.cfot_hidden is not None else cfg.get("cfot_hidden"),
    cfot_iters=args.cfot_iters if args.cfot_iters is not None else cfg.get("cfot_iters"),
    cfot_tau=args.cfot_tau if args.cfot_tau is not None else cfg.get("cfot_tau"),
    cfot_topk=args.cfot_topk if args.cfot_topk is not None else cfg.get("cfot_topk"),
    cfot_inject=args.cfot_inject if args.cfot_inject is not None else cfg.get("cfot_inject"),
    cfot_beta=args.cfot_beta if args.cfot_beta is not None else cfg.get("cfot_beta"),

    cfot_affinity=args.cfot_affinity if args.cfot_affinity is not None else cfg.get("cfot_affinity"),
    cfot_euclid_scale=args.cfot_euclid_scale if args.cfot_euclid_scale is not None else cfg.get("cfot_euclid_scale"),
    cfot_cosine_eps=args.cfot_cosine_eps if args.cfot_cosine_eps is not None else cfg.get("cfot_cosine_eps"),
    cfot_pos_weight=args.cfot_pos_weight if args.cfot_pos_weight is not None else cfg.get("cfot_pos_weight"),
    cfot_vel_weight=args.cfot_vel_weight if args.cfot_vel_weight is not None else cfg.get("cfot_vel_weight"),
    )


    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", {})

    # Backfill missing affinity/CFOT knobs from checkpoint config
    cfg = _merge_ckpt_cfg(cfg, ckpt_cfg)

    # Default CFOT affinity if still missing
    if cfg.get("cfot_affinity", None) is None:
        cfg["cfot_affinity"] = "euclid"  # explicit default

    # Seed + device
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Build dataset/loader
    # test_dir = cfg["test_dir"]
    feat = cfg.get("feat", "xyz")
    max_T = int(cfg.get("max_T", 30))
    normalize = bool(cfg.get("normalize", False))
    temporal_mode = cfg.get("temporal_mode", "interp")

    # Label map same as train

    test_ds = IndianSignSequenceDataset(
    root_split_dir=cfg["test_dir"],
    max_T=int(cfg.get("max_T", 30)),
    feat=cfg.get("feat", "xyz"),
    normalize=bool(cfg.get("normalize", True)),
    temporal_mode=cfg.get("temporal_mode", "interp"),
    aug=False,
    eval_mode=True,
    )

    batch_size = int(cfg.get("batch", 64))
    test_loader = DataLoader(
    test_ds,
    batch_size=int(cfg.get("batch", 8)),
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_sign,
)
    
    num_classes = len(test_ds.class_names)
    cfg["num_class"] = num_classes



    # Output dir
    out_root = Path(cfg.get("save_dir", "runs/isl_stgcn"))
    exp_name = cfg.get("exp_name", ckpt_path.parent.name)
    out_dir = Path(args.out_dir) if args.out_dir else (out_root / exp_name / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model with final cfg (keep architecture; do not remove CFOT)
    model = build_model(cfg, device)

    # Optionally pre-disable CFOT before loading weights
    if args.disable_cfot:
        hard_disable_cfot(model)

    # Load weights (strict=False tolerates minor shape diffs)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] Warning: {len(missing)} missing keys (OK if heads differ)")
    if unexpected:
        print(f"[eval] Warning: {len(unexpected)} unexpected keys")

    # Restore CFOT strength if recorded, then possibly override with disable flag
    if hasattr(model, "cfot_strength_frac") and ("cfot_strength_frac" in ckpt):
        try:
            model.cfot_strength_frac = float(ckpt["cfot_strength_frac"])
        except Exception:
            pass

    # Final CFOT mode selection
    if args.disable_cfot:
        hard_disable_cfot(model)
        print("[eval] CFOT forced OFF (strength=0, beta=0, affinity=euclid).")
    else:
        s = float(getattr(model, "cfot_strength_frac", 0.0))
        a = getattr(model, "cfot_affinity", cfg.get("cfot_affinity", "n/a"))
        print(f"[eval] CFOT strength fraction restored to {s:.3f}; affinity={a}.")

        # Sanity: warn if affinity differs from checkpoint config
        if "cfot_affinity" in ckpt_cfg:
            tr_aff = str(ckpt_cfg["cfot_affinity"]).lower()
            ev_aff = str(cfg.get("cfot_affinity", "euclid")).lower()
            if tr_aff != ev_aff:
                print(f"[eval][WARN] Affinity mismatch: train={tr_aff} vs eval={ev_aff}. "
                      f"Proceeding with eval={ev_aff}.")

    # Evaluate
    dump_path = out_dir / "predictions.csv" if args.dump_preds else None
    stats = run_eval(model, test_loader, device, dump_preds_path=dump_path)

    # Report
    N = len(test_ds)
    print(f"Test accuracy: {stats['acc']:.2f}%  (N={N})")
    print(f"Top-1: {stats['top1']:.2f}%  | Top-5: {stats['top5']:.2f}%")
    print("[timing] "
          f"forward_total={stats['forward_total']:.3f}s | wall_total={stats['wall_total']:.3f}s | "
          f"avg_forward={stats['avg_forward_ms']:.3f} ms/sample | avg_wall={stats['avg_wall_ms']:.3f} ms/sample | "
          f"throughput_forward={stats['throughput_forward']:.1f} samp/s | throughput_wall={stats['throughput_wall']:.1f} samp/s")

    # Save artifacts
    meta = {
        "cfg": cfg,
        "ckpt": str(ckpt_path),
        "stats": stats,
        "num_samples": N,
        "cfot_disabled": bool(args.disable_cfot),
    }
    with (out_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if dump_path is not None:
        print(f"Saved raw predictions: {dump_path}")
    print(f"Saved eval artifacts to: {out_dir}")

if __name__ == "__main__":
    main()
