#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz.py — lightweight utilities to LOG and VISUALIZE spatial & temporal connections
==================================================================================
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any, Union
import numpy as np
import torch

from matplotlib.ticker import FixedLocator, FixedFormatter

import matplotlib as mpl

mpl.use("Agg")                  # set backend BEFORE importing pyplot
import matplotlib.pyplot as plt

# networkx is optional; import lazily
try:
    import networkx as nx
except Exception:
    nx = None


# ------------------------------
# Logging (during training)
# ------------------------------

# -------------------- viz utils (fixed) --------------------
import os, math, random, numpy as np
import matplotlib.pyplot as plt

class AdjLogger:
    """
    Keeps running sums of per-delta adjacency matrices and exports:
      - epoch snapshots of spatial/temporal adjacencies
      - optional per-class 'random frame' heatmaps
    Robust to torch.Tensors and np.ndarrays.
    """
    def _to_np(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float().numpy()
        except Exception:
            pass
        x = np.asarray(x)
        return x.astype(np.float32, copy=False)

    def __init__(self, save_root, V, deltas, track_class=None):
        self.save_root = save_root
        os.makedirs(save_root, exist_ok=True)
        self.V = int(V)
        self.deltas = list(int(d) for d in deltas)  # e.g., [1]
        self.D = len(self.deltas)
        # running sums (D, V, V)
        self.spatial_sum = np.zeros((self.D, self.V, self.V), dtype=np.float32)
        self.temporal_sum = np.zeros((self.D, self.V, self.V), dtype=np.float32)
        self.spatial_count = 0
        self.temporal_count = 0
        # last raw (for saving even at epoch 0)
        self._last_spatial = np.zeros((self.D, self.V, self.V), dtype=np.float32)
        self._last_temporal = np.zeros((self.D, self.V, self.V), dtype=np.float32)
        self.track_class = track_class

    # --- internal helpers
    def _coerce_DVV(self, arr):
        """
        Accept (V,V) or (D,V,V) and return (D,V,V).
        """
        arr = self._to_np(arr)
        if arr.ndim == 2 and arr.shape == (self.V, self.V):
            # broadcast to all deltas
            arr = np.broadcast_to(arr, (self.D, self.V, self.V)).copy()
        elif arr.ndim == 3 and arr.shape[1:] == (self.V, self.V):
            assert arr.shape[0] == self.D, f"expected D={self.D}, got {arr.shape[0]}"
            arr = arr.copy()
        else:
            raise ValueError(f"adj shape must be (V,V) or (D,V,V); got {arr.shape}")
        return arr

    # --- public API used by training loop
    def update_spatial(self, raw):
        """
        raw: (V,V) or (D,V,V)
        """
        try:
            mat = self._coerce_DVV(raw)  # (D,V,V)
            self.spatial_sum += mat
            self.spatial_count += 1
            self._last_spatial = mat
        except Exception as e:
            print(f"[viz] update skipped (coerced): {e}")

    def update_temporal(self, raw):
        """
        raw: (V,V) or (D,V,V)
        """
        try:
            mat = self._coerce_DVV(raw)  # (D,V,V)
            self.temporal_sum += mat
            self.temporal_count += 1
            self._last_temporal = mat
        except Exception as e:
            print(f"[viz] update skipped (coerced): {e}")

    def _mean_or_last(self, kind):
        """
        Returns (D,V,V) matrix for 'spatial' or 'temporal'.
        Uses running mean if count>0; otherwise uses last seen raw.
        """
        if kind == "spatial":
            if self.spatial_count > 0:
                return self.spatial_sum / max(1, self.spatial_count)
            return self._last_spatial
        else:
            if self.temporal_count > 0:
                return self.temporal_sum / max(1, self.temporal_count)
            return self._last_temporal

    def _format_axes_heatmap(self, ax, V: int):
        ticks = np.arange(V)
        ax.set_xticks(ticks); ax.set_xticklabels([str(i+1) for i in ticks])
        ax.set_yticks(ticks); ax.set_yticklabels([str(i+1) for i in ticks])
        ax.set_xlim(-0.5, V-0.5); ax.set_ylim(V-0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(-0.5, V, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for sp in ax.spines.values(): sp.set_visible(False)

    def save_epoch_snapshots(self, epoch):
        """
        Save per-epoch snapshots with ticks 1..V and color scale [0,1].
        Writes:
        viz/spatial_adj_epoch_XXX.png
        viz/temporal_adj_epoch_XXX_delta_D.png  (for each Δ)
        """
        try:
            out_dir = os.path.join(self.save_root, "viz")
            os.makedirs(out_dir, exist_ok=True)

            # ---- spatial (no delta) ----
            S = self._mean_or_last("spatial")      # (D,V,V)
            A = np.clip(S.mean(axis=0), 0.0, 1.0)  # average over deltas → (V,V)
            V = A.shape[-1]
            fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
            im = ax.imshow(A, vmin=0.0, vmax=1.0, interpolation="nearest")
            fig.colorbar(im, ax=ax)
            ax.set_title(f"spatial — epoch {epoch:03d}")
            ax.set_xlabel("target joints")
            ax.set_ylabel("source joints")
            ticks = np.arange(V); labels = [str(i+1) for i in range(V)]
            ax.set_xlim(-0.5, V-0.5); ax.set_ylim(V-0.5, -0.5); ax.set_aspect("equal")
            ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels))
            ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter(labels))
            ax.set_xticks(np.arange(-0.5, V, 1), minor=True); ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            for sp in ax.spines.values(): sp.set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"spatial_adj_epoch_{epoch:03d}.png"))
            plt.close(fig)

            # ---- temporal (per delta) ----
            Tmats = self._mean_or_last("temporal")  # (D,V,V)
            for d_idx, d in enumerate(self.deltas):
                M = np.clip(np.asarray(Tmats[d_idx], dtype=np.float32), 0.0, 1.0)
                V = M.shape[-1]
                fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
                im = ax.imshow(M, vmin=0.0, vmax=1.0, interpolation="nearest")
                fig.colorbar(im, ax=ax)
                ax.set_title(f"temporal — epoch {epoch:03d} — Δ={d}")
                ax.set_xlabel("target joints")
                ax.set_ylabel("source joints")
                ticks = np.arange(V); labels = [str(i+1) for i in range(V)]
                ax.set_xlim(-0.5, V-0.5); ax.set_ylim(V-0.5, -0.5); ax.set_aspect("equal")
                ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels))
                ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter(labels))
                ax.set_xticks(np.arange(-0.5, V, 1), minor=True); ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
                ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
                ax.tick_params(which="minor", bottom=False, left=False)
                for sp in ax.spines.values(): sp.set_visible(False)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"temporal_adj_epoch_{epoch:03d}_delta_{d}.png"))
                plt.close(fig)

            print(f"[viz] Saved epoch{epoch} snapshots (spatial/temporal).")
        except Exception as e:
            print(f"[viz] epoch{epoch} export failed: {e}")


    def save_random_frame_heatmaps(self, epoch, cls_id):
        """
        Optional per-class random-frame heatmap (uses temporal mean as placeholder),
        rendered with fixed 0..1 scale and 1..V ticks.
        """
        try:
            out_dir = os.path.join(self.save_root, "viz_track")
            os.makedirs(out_dir, exist_ok=True)

            mats = self._mean_or_last("temporal")  # (D,V,V)
            A = np.clip(np.asarray(mats[0], dtype=np.float32), 0.0, 1.0)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            im = ax.imshow(A, vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"class {cls_id} — epoch {epoch:03d} — t→t+{self.deltas[0]}")
            ax.set_xlabel("target joints (t+1)")
            ax.set_ylabel("source joints (t)")
            self._format_axes_heatmap(ax, self.V)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}__idx_{random.randint(0, 3000)}.png"), dpi=160)
            plt.close(fig)
            print(f"[viz] Saved Δ={self.deltas[0]} random-frame heatmaps @ epoch {epoch} -> {out_dir}")
        except Exception as e:
            print(f"[viz] Δ heatmap export failed: {e}")
# ------------------ end viz utils (fixed) ------------------



# ------------------------------
# Hand layout utilities
# ------------------------------

def open_hand_layout(V: int = 22) -> Dict[int, Tuple[float, float]]:
    xs = np.linspace(-2.0, 2.0, 5)
    coords: Dict[int, Tuple[float, float]] = {}
    coords[0] = (0.0, -2.2)
    for i in range(1, 6):
        coords[i] = (xs[i-1] * 0.7, -1.2)
    jid = 6
    for col in range(4):
        x = xs[col+1] * 0.9
        for r in range(4):
            coords[jid] = (x, -0.6 + r * 0.6)
            jid += 1
    for j in range(V):
        coords.setdefault(j, (0.0, 0.0))
    return coords


# ------------------------------
# Plotting helpers
# ------------------------------

def _apply_ticks_and_grid(ax, V: int):
    ticks = np.arange(V)  # 0..V-1 positions
    labels = [str(i+1) for i in ticks]  # show 1..V

    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))

    ax.set_xlim(-0.5, V-0.5)
    ax.set_ylim(V-0.5, -0.5)
    ax.set_aspect("equal")

    # grid at cell boundaries
    ax.set_xticks(np.arange(-0.5, V, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def _trim_edges_by_threshold(A: np.ndarray,
                             threshold: Optional[float] = None,
                             keep_percent: float = 10.0) -> np.ndarray:
    W = A.copy()
    V = W.shape[0]
    mask = np.zeros_like(W, dtype=bool)
    if threshold is None:
        flat = W.flatten()
        k = max(1, int((keep_percent / 100.0) * flat.size))
        thr = np.partition(flat, -k)[-k]
        threshold = float(thr)
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            if W[i, j] >= threshold:
                mask[i, j] = True
    return mask


def plot_spatial_graph(A: np.ndarray,
                       save_path: Optional[Union[str, Path]] = None,
                       coords: Optional[Dict[int, Tuple[float, float]]] = None,
                       title: Optional[str] = None,
                       keep_percent: float = 10.0) -> None:
    if coords is None:
        coords = open_hand_layout(A.shape[0])
    if nx is None:
        raise ImportError("networkx is required for plot_spatial_graph")
    V = A.shape[0]
    mask = _trim_edges_by_threshold(A, threshold=None, keep_percent=keep_percent)

    G = nx.DiGraph()
    G.add_nodes_from(range(V))
    for i in range(V):
        for j in range(V):
            if mask[i, j]:
                G.add_edge(i, j, weight=float(A[i, j]))

    plt.figure()
    nx.draw_networkx_nodes(G, coords, node_size=100)
    widths = [d.get("weight", 0.0) for _, _, d in G.edges(data=True)]
    if widths:
        arr = np.array(widths, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = 0.5 + 3.0 * (arr - lo) / (hi - lo)
        else:
            arr = 1.0 + 0 * arr
        nx.draw_networkx_edges(G, coords, width=arr, arrows=True, arrowsize=10)
    nx.draw_networkx_labels(G, coords, font_size=6)

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()

def save_heatmap(A: np.ndarray, out_png: Path, title: str = "", cmap: str = "magma"):
    A = np.asarray(A, dtype=float)
    A = np.clip(A, 0.0, 1.0)             # force [0,1]
    V = A.shape[-1]

    fig, ax = plt.subplots(figsize=(4,4), dpi=200)
    im = ax.imshow(A, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")

    if title: ax.set_title(title)

    # ticks: 1..V
    ticks = np.arange(V)
    labels = [str(i+1) for i in range(V)]
    ax.set_xlim(-0.5, V-0.5)
    ax.set_ylim(V-0.5, -0.5)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))

    # gridlines between cells
    ax.set_xticks(np.arange(-0.5, V, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for sp in ax.spines.values(): sp.set_visible(False)

    ax.set_xlabel("target joint")
    ax.set_ylabel("source joint")
    fig.colorbar(im, ax=ax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)



def plot_temporal_heatmap(M: np.ndarray,
                          save_path: Optional[Union[str, Path]] = None,
                          title: Optional[str] = None) -> None:
    H = np.asarray(M, dtype=float)
    if H.ndim == 3:  # [T,V,V] → average
        H = H.mean(axis=0)
    H = np.clip(H, 0.0, 1.0)

    V = H.shape[-1]
    fig, ax = plt.subplots(figsize=(4,4), dpi=200)
    im = ax.imshow(H, vmin=0.0, vmax=1.0, interpolation="nearest")

    if title: ax.set_title(title)

    ticks = np.arange(V)
    labels = [str(i+1) for i in range(V)]
    ax.set_xlim(-0.5, V-0.5)
    ax.set_ylim(V-0.5, -0.5)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))

    ax.set_xticks(np.arange(-0.5, V, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, V, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for sp in ax.spines.values(): sp.set_visible(False)

    ax.set_xlabel("target joint")
    ax.set_ylabel("source joint")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)




def plot_per_delta_heatmaps(B: np.ndarray,
                            deltas: Optional[Sequence[int]] = None,
                            save_dir: Union[str, Path] = ".",
                            epoch: Optional[int] = None,
                            prefix: str = "viz_temporal_d") -> None:
    """Plot per-delta heatmaps from [D,V,V] or [T,D,V,V] with fixed 0..1 scale and 1..V ticks."""
    save_dir = Path(save_dir)
    H = np.asarray(B, dtype=np.float32)
    if H.ndim == 4:  # [T,D,V,V] → [D,V,V]
        H = H.mean(axis=0)
    H = np.clip(H, 0.0, 1.0)
    D = H.shape[0]
    for d in range(D):
        name = f"{prefix}{d}"
        if deltas is not None and d < len(deltas):
            name += f"_Δ{deltas[d]}"
        if epoch is not None:
            name += f"_epoch{epoch}"
        outp = save_dir / f"{name}.png"
        plot_temporal_heatmap(H[d], save_path=outp, title=f"Temporal (Δ index {d})")


def plot_evolution_spatial(run_dir: Union[str, Path],
                           epochs: Sequence[int],
                           coords: Optional[Dict[int, Tuple[float, float]]] = None,
                           keep_percent: float = 10.0) -> List[Path]:
    run_dir = Path(run_dir)
    outs: List[Path] = []
    for e in epochs:
        p = run_dir / f"spatial_adj_epoch{e}.npy"
        if p.exists():
            A = np.load(p)
            outp = run_dir / f"viz_spatial_epoch{e}.png"
            plot_spatial_graph(A, save_path=outp, coords=coords, title=f"Spatial — epoch {e}", keep_percent=keep_percent)
            outs.append(outp)
    return outs


def plot_evolution_temporal(run_dir: Union[str, Path],
                            epochs: Sequence[int]) -> List[Path]:
    run_dir = Path(run_dir)
    outs: List[Path] = []
    for e in epochs:
        p = run_dir / f"temporal_adj_epoch{e}.npy"
        if p.exists():
            T = np.load(p)
            outp = run_dir / f"viz_temporal_epoch{e}.png"
            plot_temporal_heatmap(T, save_path=outp, title=f"Temporal (avg) — epoch {e}")
            outs.append(outp)
    return outs

@torch.no_grad()
def export_classwise_cfot_heatmaps(model,
                                   dataset,
                                   device,
                                   out_dir: Union[str, Path],
                                   epoch: int,
                                   max_per_class: int = 0,
                                   use_per_delta: bool = True,
                                   cmap: str = "magma"):
    """
    Saves classwise CFOT heatmaps; fixed 0..1 scale and 1..V ticks.
    """
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=getattr(dataset, "collate_fn", None) or (lambda b: b[0]))

    sums_avg = {}
    counts = {}
    sums_by_delta = {}

    for sample in dl:
        if isinstance(sample, dict):
            x = sample["data"].to(device, non_blocking=True)
            y = int(sample["label"])
        else:
            x, y = sample
            x = x.to(device); y = int(y)

        if max_per_class > 0 and counts.get(y, 0) >= max_per_class:
            continue

        _ = model(x)
        Aavg = getattr(model, "cfot_last_avg_adj", None)
        AbyD = getattr(model, "cfot_last_adj_by_delta", None)

        if isinstance(Aavg, torch.Tensor):
            Aavg = Aavg.detach().float().cpu().numpy()
            sums_avg[y] = sums_avg.get(y, 0) + Aavg
            counts[y] = counts.get(y, 0) + 1

        if use_per_delta and isinstance(AbyD, torch.Tensor):
            AbyD = AbyD.detach().float().cpu().numpy()  # [D,V,V]
            if y not in sums_by_delta:
                sums_by_delta[y] = AbyD.copy()
            else:
                sums_by_delta[y] += AbyD
            counts[y] = counts.get(y, 0)

    out_dir = Path(out_dir) / "classes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for c, n in counts.items():
        if n <= 0:
            continue
        cls_dir = out_dir / str(c)
        cls_dir.mkdir(parents=True, exist_ok=True)

        if c in sums_avg:
            A = np.clip((sums_avg[c] / float(n)).astype(np.float32), 0.0, 1.0)
            np.save(cls_dir / f"temporal_adj_epoch{epoch}.npy", A)
            save_heatmap(A, cls_dir / f"temporal_adj_epoch{epoch}.png", f"class {c} — epoch {epoch}", cmap=cmap)

        if use_per_delta and c in sums_by_delta:
            BD = np.clip((sums_by_delta[c] / float(n)).astype(np.float32), 0.0, 1.0)
            np.save(cls_dir / f"temporal_by_delta_epoch{epoch}.npy", BD)
            for d in range(BD.shape[0]):
                save_heatmap(BD[d], cls_dir / f"temporal_d{d}_epoch{epoch}.png", f"class {c} Δ{d} — epoch {epoch}", cmap=cmap)



def delta_usage_bar(B: np.ndarray,
                    deltas: Optional[Sequence[int]] = None,
                    save_path: Optional[Union[str, Path]] = None,
                    title: Optional[str] = "Δ usage (mass per delta)") -> None:
    H = np.asarray(B)
    if H.ndim == 4:  # [T,D,V,V]
        H = H.sum(axis=0)  # [D,V,V]
    mass = H.sum(axis=(1, 2))  # [D]
    if mass.sum() > 0:
        mass = mass / mass.sum()
    x = np.arange(len(mass))
    labels = [f"Δ{d}" if deltas is not None and i < len(deltas) else str(i) for i, d in enumerate(deltas or [])]
    if not labels:
        labels = [str(i) for i in range(len(mass))]

    plt.figure()
    plt.bar(x, mass)
    plt.xticks(x, labels, rotation=0)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
