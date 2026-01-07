"""
utils/normalize.py
-------------------
Lightweight preprocessing helpers shared across datasets.

All functions assume skeleton sequences shaped as [T, V, 3]
unless stated otherwise.

Provided helpers
- root_center_skeleton(seq, root_joint=0)
- unit_bone_scale(seq, pairs=None, eps=1e-6)
- linear_time_resample(seq, L)
- add_velocity_channel(seq_xyz)
- get_bone_pairs(layout="shrec22")

Notes
- Velocity is computed as v[t] = x[t] - x[t-1] with v[0]=0.
- linear_time_resample uses per-dimension 1D interpolation.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

__all__ = [
    "root_center_skeleton",
    "unit_bone_scale",
    "linear_time_resample",
    "add_velocity_channel",
    "get_bone_pairs",
]


# ------------------------------
# Bone layouts
# ------------------------------

def _bone_pairs_shrec22() -> List[Tuple[int, int]]:
    """A simple 22-joint hand kinematic chain (SHREC-like).
   # 0: wrist center (assumed root)
        # Then fingers: thumb(2-5), index(6-9), middle(10-13), ring(14-17), pinky(18-21)
    """
    pairs = [
            (0, 1),  
            (0, 2),  (2, 3),  (3, 4),  (4, 5), 
            (1, 6),  (6, 7),  (7, 8),  (8, 9), 
            (1, 10), (10, 11),  (11, 12),  (12, 13), 
            (1, 14), (14, 15),  (15, 16),  (16, 17),  
            (1, 18),  (18, 19),  (19, 20),  (20, 21), ]
    return pairs


def _bone_pairs_mediapipe21() -> List[Tuple[int, int]]:
    """MediaPipe Hands (21) canonical bones (0 wrist).
    This is provided for completeness; not used by SHREC'17 loader by default.
    """
    return [
        (0,1),(1,2),(2,3),(3,4),            # thumb
        (0,5),(5,6),(6,7),(7,8),            # index
        (0,9),(9,10),(10,11),(11,12),       # middle
        (0,13),(13,14),(14,15),(15,16),     # ring
        (0,17),(17,18),(18,19),(19,20),     # pinky
    ]




def get_bone_pairs(layout: str = "shrec22") -> List[Tuple[int, int]]:
    layout = layout.lower()
    if layout in ("shrec", "shrec17"):
        return _bone_pairs_shrec22()
    if layout in ("mediapipe", "mediapipe21","ipn","briareo", "21"):
        return _bone_pairs_mediapipe21()
    
    raise ValueError(f"Unknown bone layout: {layout}")


# ------------------------------
# Core transforms
# ------------------------------

def root_center_skeleton(seq: np.ndarray, root_joint: int = 0) -> np.ndarray:
    """Root-center per frame by subtracting the root joint.

    Args:
        seq: [T, V, 3]
        root_joint: index of the root joint to center at
    Returns:
        [T, V, 3] root-centered sequence
    """
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError(f"Expected [T,V,3], got {seq.shape}")
    root = seq[:, root_joint:root_joint+1, :]  # [T,1,3]
    return seq - root


def unit_bone_scale(
    seq: np.ndarray,
    pairs: Optional[List[Tuple[int, int]]] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Scale so the mean bone length across frames equals 1.0.

    Args:
        seq: [T, V, 3]
        pairs: bone pairs; if None, use SHREC22 layout
        eps: small epsilon to avoid division by zero
    Returns:
        [T, V, 3] scaled sequence
    """
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError(f"Expected [T,V,3], got {seq.shape}")

    if pairs is None:
        # pairs = _bone_pairs_shrec22()
        pairs = _bone_pairs_mediapipe21()

    # Compute lengths for all bones across time
    lengths = []
    for a, b in pairs:
        d = seq[:, a, :] - seq[:, b, :]
        l = np.sqrt(np.sum(d * d, axis=-1) + eps)  # [T]
        lengths.append(l)
    all_len = np.concatenate(lengths, axis=0)
    # Guard against degenerate cases
    mask = all_len > eps
    mean_len = np.mean(all_len[mask]) if np.any(mask) else 1.0
    if mean_len < eps:
        mean_len = 1.0
    return seq / float(mean_len)


def linear_time_resample(seq: np.ndarray, L: int) -> np.ndarray:
    """Linearly resample sequence along time to length L.

    Args:
        seq: [T, V, 3]
        L: desired output length
    Returns:
        [L, V, 3]
    """
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError(f"Expected [T,V,3], got {seq.shape}")

    T, V, C = seq.shape
    if L <= 0:
        raise ValueError("L must be positive")
    if T == L:
        return seq
    if T < 2:
        # Not enough frames to interpolate — repeat the single frame
        return np.repeat(seq, L, axis=0)

    # Vectorized interpolation: reshape to [T, V*C]
    x = np.linspace(0.0, T - 1.0, num=T, dtype=np.float32)
    xp = np.linspace(0.0, T - 1.0, num=L, dtype=np.float32)
    flat = seq.reshape(T, V * C)
    out = np.empty((L, V * C), dtype=seq.dtype)
    for d in range(V * C):
        out[:, d] = np.interp(xp, x, flat[:, d])
    return out.reshape(L, V, C)


def add_velocity_channel(seq_xyz: np.ndarray) -> np.ndarray:
    """Concatenate velocity channels to xyz.

    Args:
        seq_xyz: [T, V, 3]
    Returns:
        [T, V, 6] with channels [x,y,z,vx,vy,vz]
    """
    if seq_xyz.ndim != 3 or seq_xyz.shape[2] != 3:
        raise ValueError(f"Expected [T,V,3], got {seq_xyz.shape}")

    vel = np.zeros_like(seq_xyz)
    if seq_xyz.shape[0] > 1:
        vel[1:] = seq_xyz[1:] - seq_xyz[:-1]
        # keep vel[0] = 0
    return np.concatenate([seq_xyz, vel], axis=2)
