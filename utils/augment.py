"""
utils/augment.py (regenerated)
------------------------------
Lightweight 3D hand-skeleton augmentations + temporal sampling helpers.

All functions assume input shape **[T, V, 3]** and return the same shape
(unless noted). Designed to be fast, numpy-only, and numerically stable.

Provided API (stable names used by the loaders):
- random_rotate(seq, max_deg=15.0, axes=("x","y","z"), center=None)
- random_scale(seq, scale_range=(0.9, 1.1), anisotropic_prob=0.3)
- random_jitter(seq, sigma=0.01, clip=0.05)
- random_time_flip(seq, p=0.0)
- temporal_indices(T, L, mode='crop_repeat', train=True)

Extras (not yet wired by default, but ready to use):
- random_translate(seq, sigma=0.01)
- time_warp_indices(T, L, strength=0.15, train=True)

Notes
- Augmentations are intended to run **after** root-centering and unit-bone
  scaling. If your data is not centered, pass `center` to random_rotate.
- Dtypes are preserved (float32/float16). Internals may upcast to float32 and
  then cast back to the original dtype to avoid numerical issues.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Optional
import numpy as np

__all__ = [
    "random_rotate",
    "random_scale",
    "random_jitter",
    "random_time_flip",
    "temporal_indices",
    # extras
    "random_translate",
    "time_warp_indices",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_float(seq: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
    """Return (float32_view, original_dtype)."""
    if not isinstance(seq, np.ndarray):
        seq = np.asarray(seq)
    orig = seq.dtype
    if seq.dtype != np.float32:
        seq = seq.astype(np.float32)
    return seq, orig


def _restore_dtype(seq: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return seq.astype(dtype, copy=False)


# -----------------------------------------------------------------------------
# Geometric augmentations
# -----------------------------------------------------------------------------

def random_rotate(
    seq: np.ndarray,
    max_deg: float = 15.0,
    axes: Iterable[str] = ("x", "y", "z"),
    center: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Apply a small random 3D rotation around selected axes.

    Args:
        seq: [T, V, 3]
        max_deg: maximum absolute rotation per axis (degrees)
        axes: subset of {"x","y","z"}
        center: optional (3,) center to rotate around. If None, no re-centering
                is performed (OK if input is already root-centered).
    """
    assert seq.ndim == 3 and seq.shape[2] == 3, f"expected [T,V,3], got {seq.shape}"
    x, orig = _as_float(seq)

    # Compute random rotation angles
    use_x = "x" in axes
    use_y = "y" in axes
    use_z = "z" in axes
    ax = np.deg2rad(np.random.uniform(-max_deg, max_deg)) if use_x else 0.0
    ay = np.deg2rad(np.random.uniform(-max_deg, max_deg)) if use_y else 0.0
    az = np.deg2rad(np.random.uniform(-max_deg, max_deg)) if use_z else 0.0

    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]], dtype=np.float32)
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]], dtype=np.float32)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]], dtype=np.float32)

    # Compose in Z * Y * X order (conventional)
    R = Rz @ Ry @ Rx

    # Optional centering around a point
    if center is not None:
        c = np.asarray(center, dtype=np.float32)[None, None, :]  # [1,1,3]
        x = x - c
        y = (x.reshape(-1, 3) @ R.T).reshape(seq.shape)
        y = y + c
    else:
        y = (x.reshape(-1, 3) @ R.T).reshape(seq.shape)

    return _restore_dtype(y, orig)


def random_scale(
    seq: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    anisotropic_prob: float = 0.3,
) -> np.ndarray:
    """Random isotropic or anisotropic scaling.

    If anisotropic is chosen, each axis gets an independent factor sampled
    from `scale_range`. Otherwise a single scalar is used for all axes.
    """
    assert seq.ndim == 3 and seq.shape[2] == 3
    x, orig = _as_float(seq)
    lo, hi = scale_range
    lo, hi = float(lo), float(hi)

    if np.random.rand() < float(anisotropic_prob):
        s = np.random.uniform(lo, hi, size=(3,)).astype(np.float32)  # (sx,sy,sz)
    else:
        s_val = float(np.random.uniform(lo, hi))
        s = np.array([s_val, s_val, s_val], dtype=np.float32)

    y = x * s[None, None, :]
    return _restore_dtype(y, orig)


def random_translate(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Small global translation noise (useful if not root-centered)."""
    assert seq.ndim == 3 and seq.shape[2] == 3
    x, orig = _as_float(seq)
    t = np.random.normal(0.0, sigma, size=(1, 1, 3)).astype(np.float32)
    y = x + t
    return _restore_dtype(y, orig)


def random_jitter(seq: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """Add per-joint Gaussian noise (clipped)."""
    assert seq.ndim == 3 and seq.shape[2] == 3
    x, orig = _as_float(seq)
    noise = np.clip(np.random.normal(0.0, sigma, size=x.shape), -clip, clip).astype(np.float32)
    y = x + noise
    return _restore_dtype(y, orig)


def random_time_flip(seq: np.ndarray, p: float = 0.0) -> np.ndarray:
    """Reverse sequence order with probability p."""
    if p > 0.0 and np.random.rand() < p:
        return seq[::-1].copy()
    return seq


# -----------------------------------------------------------------------------
# Temporal sampling
# -----------------------------------------------------------------------------

def temporal_indices(
    T: int,
    L: int,
    mode: str = "crop_repeat",
    train: bool = True,
) -> np.ndarray:
    """Return integer indices of length L sampled from range [0, T-1].

    modes:
      - 'crop_repeat' (recommended):
          * if T >= L: contiguous window; random start (train) or center (eval)
          * if T <  L: tile frames to >= L, then random circular shift (train)
      - 'interp' : not index-based; included for API symmetry (handled upstream)
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    if mode == "interp":
        # Upstream code should call a resampler; provide evenly-spaced indices
        return np.linspace(0, T - 1, num=L, dtype=int)

    if mode != "crop_repeat":
        raise ValueError(f"Unknown temporal mode: {mode}")

    if T >= L:
        if train:
            start = 0 if T == L else np.random.randint(0, T - L + 1)
        else:
            start = max(0, (T - L) // 2)
        return np.arange(start, start + L, dtype=int)
    else:
        reps = int(np.ceil(L / T))
        idx = np.tile(np.arange(T, dtype=int), reps)[:L]
        if train and L > T:
            shift = int(np.random.randint(0, T))
            idx = np.roll(idx, shift)
        return idx


# -----------------------------------------------------------------------------
# Optional temporal warping (not used by default)
# -----------------------------------------------------------------------------

def time_warp_indices(T: int, L: int, strength: float = 0.15, train: bool = True) -> np.ndarray:
    """Non-linear time warping indices (smooth monotonic mapping).

    Args:
        T: original length
        L: target length
        strength: warping intensity in [0,1], where 0=no warp, 0.15~mild
        train: add randomness only in training
    """
    if not train or strength <= 0.0:
        return np.linspace(0, T - 1, num=L, dtype=int)

    # Base grid in [0,1]
    t = np.linspace(0.0, 1.0, num=L, dtype=np.float32)

    # Random smooth displacement using low-frequency sine mixture
    f1, f2 = np.random.uniform(0.5, 1.5), np.random.uniform(1.5, 3.0)
    phase1, phase2 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)
    disp = (np.sin(2*np.pi*f1*t + phase1) + 0.5*np.sin(2*np.pi*f2*t + phase2))
    disp = (disp - disp.mean()) / (np.abs(disp).max() + 1e-6)
    t_warp = np.clip(t + strength * disp, 0.0, 1.0)

    # Ensure monotonicity by cumulative maximum (keeps order)
    t_warp = np.maximum.accumulate(t_warp)

    # Map to indices
    idx = np.floor(t_warp * (T - 1) + 0.5).astype(int)
    idx = np.clip(idx, 0, T - 1)
    return idx
