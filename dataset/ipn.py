"""
IPN Hand Gesture Dataset Loader (21 joints) with OFFLINE augmentation
---------------------------------------------------------------------

Output tensor: [C, T, V, M] with V=21, M=1.

Key new args:
  - offline_aug_factor: int >= 0. Add this many cached augmented copies per sample.
  - offline_cache_dir: str/Path. Folder where augmented .npy are saved.
  - build_cache: bool. If True, missing aug files are created on first access.

Notes:
  * For training with offline augmentation: set offline_aug_factor>0, offline_cache_dir="cache/ipn_offline", build_cache=True.
  * For validation/test: keep offline_aug_factor=0, aug=False, eval_mode=True.
"""

from __future__ import annotations
import os
import re
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

# ------------------------------
# Optional shared utilities
# ------------------------------
try:
    from utils.normalize import (
        root_center_skeleton,
        unit_bone_scale,
        add_velocity_channel,
        linear_time_resample,
    )
    _HAS_UTILS = True
except Exception:
    _HAS_UTILS = False

# Local fallbacks
def _linear_time_resample(seq: np.ndarray, L: int) -> np.ndarray:
    T, V, C = seq.shape
    if T == L: return seq
    if T < 2:  return np.repeat(seq, L, axis=0)
    x  = np.linspace(0, T-1, T, dtype=np.float32)
    xp = np.linspace(0, T-1, L, dtype=np.float32)
    out = np.empty((L, V, C), dtype=seq.dtype)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(xp, x, seq[:, v, c])
    return out

def _root_center_skeleton(seq: np.ndarray, root_joint: int = 0) -> np.ndarray:
    root = seq[:, root_joint:root_joint+1, :]
    return seq - root

_HAND_PAIRS_21: List[Tuple[int,int]] = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

def _unit_bone_scale_21(seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lens = []
    for a,b in _HAND_PAIRS_21:
        d = seq[:, a, :] - seq[:, b, :]
        l = np.sqrt((d*d).sum(-1) + eps)
        lens.append(l)
    mean_bone = max(float(np.concatenate(lens,0).mean()), eps)
    return seq / mean_bone

def _add_velocity_channel(seq_xyz: np.ndarray) -> np.ndarray:
    T, V, _ = seq_xyz.shape
    vel = np.zeros_like(seq_xyz)
    vel[1:] = seq_xyz[1:] - seq_xyz[:-1]
    return np.concatenate([seq_xyz, vel], axis=2)  # [T,V,6]

# ------------------------------
# Optional online aug utilities
# ------------------------------
try:
    from utils.augment import (
        random_rotate,
        random_scale,
        random_jitter,
        temporal_indices,
    )
    _HAS_AUG = True
except Exception:
    _HAS_AUG = False

# Deterministic temporal indices
def _temporal_indices(T: int, L: int, mode: str = "crop_repeat", train: bool = True) -> np.ndarray:
    if mode == "interp":
        return np.linspace(0, T-1, num=L, dtype=int)
    if T >= L:
        start = 0 if (not train or T == L) else np.random.randint(0, T-L+1)
        return np.arange(start, start+L, dtype=int)
    reps = int(np.ceil(L / T))
    idx = np.tile(np.arange(T, dtype=int), reps)[:L]
    if train and L > T:
        shift = np.random.randint(0, T)
        idx = np.roll(idx, shift)
    return idx

# ------------------------------
# File IO (IPN txt)
# ------------------------------
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def _resolve_path(data_dir: Optional[str], folder: str, seq_id: str, gesture: str,
                  s: str, e: str, extra: str) -> str:
    fname = f"{seq_id}_{gesture}_{s}_{e}_{extra}.txt"
    return os.path.join(data_dir if data_dir else "", folder, fname)

def _load_ipn_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        blocks = f.read().split("\n\n")

    frames = []
    for blk in blocks:
        if not blk.strip():
            continue
        rows = blk.strip().split("\n")
        pts = []
        for line in rows[:21]:
            s = line.replace(";", " ").replace(",", " ").strip()
            if not s:
                pts.append([-1.0, -1.0, -1.0]); continue
            nums = [float(m.group(0)) for m in _FLOAT_RE.finditer(s)]
            if len(nums) < 3:
                pts.append([-1.0, -1.0, -1.0])
            else:
                x, y, z = nums[:3]
                if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                    x, y, z = -1.0, -1.0, -1.0
                pts.append([x, y, z])
        while len(pts) < 21:
            pts.append([-1.0, -1.0, -1.0])
        arr = np.asarray(pts[:21], dtype=np.float32)  # (21,3)
        if np.isfinite(arr).sum() >= 3:
            frames.append(arr)

    if len(frames) == 0:
        raise ValueError(f"No valid frames in {path}")
    if len(frames) == 1:
        frames.append(frames[0].copy())
    return np.stack(frames, axis=0)  # [T,21,3]

# ------------------------------
# CSV parsing
# ------------------------------
def read_ipn_annotation_csv(ann_path: str, data_dir: Optional[str]) -> List[Tuple[str,int]]:
    """
    CSV line: folder, seq_id, gesture, s, e, extra, ...
    Label = int(gesture) - 1.
    """
    items: List[Tuple[str,int]] = []
    with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = [p.strip() for p in raw.strip().split(",")]
            if len(parts) < 6:
                continue
            folder, seq_id, gesture, s, e, extra = parts[:6]
            label = int(gesture) - 1
            p = _resolve_path(data_dir, folder, seq_id, gesture, s, e, extra)
            if not os.path.exists(p):
                if seq_id == 'D0X':
                    continue
                continue
            items.append((p, label))
    if not items:
        raise RuntimeError(f"No valid samples in {ann_path}")
    return items

def build_ipn_label_map(train_csv: str, data_dir: str) -> dict:
    items = read_ipn_annotation_csv(train_csv, data_dir)
    raw = sorted({y for _, y in items})
    return {y:i for i,y in enumerate(raw)}  # raw → 0..K-1

# ------------------------------
# Deterministic OFFLINE augmentation
# ------------------------------
def _hash_seed(path: str, aug_id: int, salt: int = 12345) -> int:
    s = f"{path}::{aug_id}::{salt}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1)

def _aug_deterministic(seq_xyz: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply small geometric noise with fixed params derived from seed.
    Uses utils.augment if present, else simple local transforms.
    """
    rng = np.random.RandomState(seed)
    out = seq_xyz.copy()

    # rotation around Z (hand plane), small tilt X/Y
    if rng.rand() < 0.8:
        deg_z = rng.uniform(-15.0, 15.0)
        deg_x = rng.uniform(-7.0, 7.0)
        deg_y = rng.uniform(-7.0, 7.0)
        rad = np.deg2rad([deg_x, deg_y, deg_z])
        cx, sx = np.cos(rad[0]), np.sin(rad[0])
        cy, sy = np.cos(rad[1]), np.sin(rad[1])
        cz, sz = np.cos(rad[2]), np.sin(rad[2])
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
        R = (Rz @ Ry @ Rx).astype(np.float32)
        out = out @ R.T

    # global scale
    if rng.rand() < 0.8:
        s = rng.uniform(0.9, 1.1)
        out = out * s

    # jitter
    if rng.rand() < 0.5:
        noise = rng.normal(loc=0.0, scale=0.01, size=out.shape).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        out = out + noise

    return out

def _cache_file_path(cache_dir: Path, base_path: str, aug_id: int) -> Path:
    # stable short name from base path
    stem = hashlib.blake2b(base_path.encode("utf-8"), digest_size=8).hexdigest()
    return cache_dir / f"{stem}_aug{aug_id}.npy"

# ------------------------------
# Dataset
# ------------------------------
class IPNDataset(Dataset):
    """
    Output dict:
      data:  [C,T,V,1]  with V=21
      label: LongTensor
      meta:  dict(path,T,V,aug_id)
    """
    def __init__(
        self,
        ann_file: str,
        data_dir: Optional[str] = None,
        max_T: int = 80,
        normalize: bool = True,
        feat: str = "xyz",               # {"xyz","xyz+vel"}
        root_joint: int = 0,
        temporal_mode: str = "interp",   # {"crop_repeat","interp"}
        aug: bool = False,               # online aug (kept for compatibility; not used when offline_aug_factor>0)
        eval_mode: bool = False,
        label_map: Optional[dict] = None,
        dtype: str = "float32",
        # NEW: offline augmentation controls
        offline_aug_factor: int = 0,     # number of extra augmented copies per sample
        offline_cache_dir: Optional[str] = None,
        build_cache: bool = False,
    ) -> None:
        super().__init__()
        self.ann_file = ann_file
        self.data_dir = data_dir
        self.max_T = int(max_T)
        self.normalize_flag = bool(normalize)
        self.feat = str(feat).lower()
        self.root_joint = int(root_joint)
        self.temporal_mode = temporal_mode
        self.aug = bool(aug)
        self.eval_mode = bool(eval_mode)
        self.dtype = np.float32 if dtype == "float32" else np.float16

        # offline aug
        self.offline_aug_factor = max(0, int(offline_aug_factor))
        self.cache_dir = Path(offline_cache_dir) if offline_cache_dir else None
        self.build_cache = bool(build_cache)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # read items and filter
        raw_items = read_ipn_annotation_csv(ann_file, data_dir)  # (path, raw_label)
        base_items: List[Tuple[str, int]] = []
        for p, y in raw_items:
            if label_map is not None:
                if y not in label_map:
                    continue
                y = label_map[y]
            try:
                _ = _load_ipn_txt(p)  # validate once
                base_items.append((p, y))
            except Exception:
                continue

        self.num_classes = (len(label_map) if label_map is not None
                            else (max(y for _,y in base_items)+1))
        if not base_items:
            raise RuntimeError("No valid IPN samples after filtering.")

        # expand with offline aug: aug_id = 0 is original, 1..K are augmented copies
        self.items_aug: List[Tuple[str,int,int]] = []
        K = self.offline_aug_factor if not self.eval_mode else 0
        for (p, y) in base_items:
            self.items_aug.append((p, y, 0))
            for k in range(1, K+1):
                self.items_aug.append((p, y, k))

        self.V = 21
        self.M = 1

    def __len__(self) -> int:
        return len(self.items_aug)

    # ----- core transforms pipeline -----
    def _temporal_sample(self, seq_xyz: np.ndarray) -> np.ndarray:
        T = seq_xyz.shape[0]
        if self.max_T is None or self.max_T <= 0:
            return seq_xyz
        if self.temporal_mode == "interp":
            if _HAS_UTILS:
                return linear_time_resample(seq_xyz, self.max_T)
            return _linear_time_resample(seq_xyz, self.max_T)
        idx = (temporal_indices if _HAS_AUG else _temporal_indices)(
            T, self.max_T, mode="crop_repeat", train=not self.eval_mode
        )
        return seq_xyz[idx]

    def _normalize_xyz(self, seq_xyz: np.ndarray) -> np.ndarray:
        if not self.normalize_flag:
            return seq_xyz
        seq = (_HAS_UTILS and root_center_skeleton or _root_center_skeleton)(seq_xyz, self.root_joint)
        seq = (_HAS_UTILS and unit_bone_scale or _unit_bone_scale_21)(seq)
        return seq

    def _online_augment(self, seq_xyz: np.ndarray) -> np.ndarray:
        # Kept for backward compatibility when offline_aug_factor=0
        if self.eval_mode or not self.aug or not _HAS_AUG:
            return seq_xyz
        if np.random.rand() < 0.5:
            seq_xyz = random_rotate(seq_xyz, max_deg=15.0)
        if np.random.rand() < 0.5:
            seq_xyz = random_scale(seq_xyz, (0.9, 1.1))
        if np.random.rand() < 0.3:
            seq_xyz = random_jitter(seq_xyz, sigma=0.01, clip=0.05)
        return seq_xyz

    def _to_tensor_CTVM(self, seq_xyz: np.ndarray) -> torch.Tensor:
        if self.feat == "xyz+vel":
            xv = (_HAS_UTILS and add_velocity_channel or _add_velocity_channel)(seq_xyz)  # [T,V,6]
            data = np.transpose(xv, (2, 0, 1))  # [C,T,V]
        elif self.feat == "xyz":
            data = np.transpose(seq_xyz, (2, 0, 1))  # [C,T,V]
        else:
            raise ValueError(f"Unsupported feat={self.feat}")
        data = data[..., np.newaxis]  # [C,T,V,1]
        return torch.from_numpy(data)

    # ----- offline cache helpers -----
    def _load_or_build_offline_aug(self, base_path: str, aug_id: int) -> np.ndarray:
        """
        Returns a normalized, temporally-aligned augmented sequence [T,V,3].
        Cache file stores the sequence AFTER normalization but BEFORE feature assembly.
        """
        assert aug_id >= 1
        if self.cache_dir is None:
            # no cache dir provided -> generate deterministic aug in memory
            seq = _load_ipn_txt(base_path)                 # [T,21,3]
            seq = self._temporal_sample(seq)
            seq = self._normalize_xyz(seq)
            seed = _hash_seed(base_path, aug_id)
            seq = _aug_deterministic(seq, seed)
            return seq.astype(self.dtype)

        cache_file = _cache_file_path(self.cache_dir, base_path, aug_id)
        if cache_file.exists():
            arr = np.load(str(cache_file))                 # [T,V,3]
            return arr.astype(self.dtype)

        # build if allowed
        if not self.build_cache:
            # fallback: deterministic in-memory aug without saving
            seq = _load_ipn_txt(base_path)
            seq = self._temporal_sample(seq)
            seq = self._normalize_xyz(seq)
            seed = _hash_seed(base_path, aug_id)
            seq = _aug_deterministic(seq, seed)
            return seq.astype(self.dtype)

        # create and save
        seq = _load_ipn_txt(base_path)
        seq = self._temporal_sample(seq)
        seq = self._normalize_xyz(seq)
        seed = _hash_seed(base_path, aug_id)
        seq = _aug_deterministic(seq, seed)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_file), seq.astype(np.float32))
        return seq.astype(self.dtype)

    # ----- I/O -----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_path, label, aug_id = self.items_aug[idx]

        if aug_id == 0:
            # original sample path
            seq = _load_ipn_txt(base_path)
            seq = self._temporal_sample(seq)
            seq = self._normalize_xyz(seq)
            # optional online augmentation when no offline augmentation used
            if self.offline_aug_factor == 0:
                seq = self._online_augment(seq)
        else:
            # offline-augmented sample
            seq = self._load_or_build_offline_aug(base_path, aug_id)

        data = self._to_tensor_CTVM(seq.astype(self.dtype))
        y = torch.tensor(label, dtype=torch.long)
        meta = {"path": base_path, "T": int(seq.shape[0]), "V": 21, "aug_id": int(aug_id)}
        return {"data": data, "label": y, "meta": meta}

# ------------------------------
# Collate
# ------------------------------
def ipn_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    datas  = torch.stack([b["data"] for b in batch], dim=0)  # [N,C,T,V,1]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    metas  = [b["meta"] for b in batch]
    return {"data": datas, "label": labels, "meta": metas}

# ------------------------------
# Optional pre-builder
# ------------------------------
def build_offline_cache(ann_file: str, data_dir: str, cache_dir: str,
                        max_T: int = 80, normalize: bool = True,
                        offline_aug_factor: int = 2) -> None:
    """
    Pre-generate all offline augmented copies to disk.
    """
    ds = IPNDataset(
        ann_file=ann_file,
        data_dir=data_dir,
        max_T=max_T,
        normalize=normalize,
        feat="xyz",
        aug=False,
        eval_mode=False,
        offline_aug_factor=offline_aug_factor,
        offline_cache_dir=cache_dir,
        build_cache=True,
    )
    # Iterate once to populate cache
    for i in range(len(ds)):
        _ = ds[i]  # triggers creation if missing

# ------------------------------
# Self-test
# ------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_T", type=int, default=80)
    ap.add_argument("--temporal_mode", default="interp", choices=["crop_repeat","interp"])
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--feat", default="xyz", choices=["xyz","xyz+vel"])
    ap.add_argument("--offline_aug_factor", type=int, default=0)
    ap.add_argument("--offline_cache_dir", type=str, default=None)
    ap.add_argument("--build_cache", action="store_true")
    args = ap.parse_args()

    ds = IPNDataset(
        ann_file=args.ann,
        data_dir=args.data_dir,
        max_T=args.max_T,
        temporal_mode=args.temporal_mode,
        normalize=not args.no_normalize,
        feat=args.feat,
        aug=False,                   # use offline instead
        eval_mode=False,
        offline_aug_factor=args.offline_aug_factor,
        offline_cache_dir=args.offline_cache_dir,
        build_cache=args.build_cache,
    )
    print(f"Dataset size: {len(ds)}  (factor {args.offline_aug_factor} -> x{1+max(0,args.offline_aug_factor)})")
    ex = ds[0]
    print("Sample tensor:", ex["data"].shape, "(expect [C,T,V,1])")
    print("Label:", ex["label"].item(), "Meta:", ex["meta"])
