# indian_sign_dataset_v2.py
"""
Dataset loader for your MediaPipe-extracted frames saved as:
  <root>/<action_label>/<sequence_folder>/<frame_number>.npy

Example:
  D:\Indian Sign languge Recognition\dataset\train_test\train\action_01\seq_0001\0001.npy

This loader:
 - Scans a root split directory (train/validation/test) and builds items per sequence folder.
 - Loads sequences by stacking per-frame .npy files (robust to several per-frame array shapes).
 - Supports temporal resampling, normalization (root-center + unit bone scale heuristic),
   online augmentation, deterministic offline augmentation with cache, and printing class names.
 - Outputs tensors shaped [C, T, V, 1] where C=3 (xyz) or 6 (xyz+vel).
"""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import re

# -------------------------
# Helpers
# -------------------------
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

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
    root = seq[:, root_joint:root_joint+1, :].copy()
    return seq - root

def _add_velocity_channel(seq_xyz: np.ndarray) -> np.ndarray:
    T, V, _ = seq_xyz.shape
    vel = np.zeros_like(seq_xyz)
    vel[1:] = seq_xyz[1:] - seq_xyz[:-1]
    return np.concatenate([seq_xyz, vel], axis=2)  # [T,V,6]

def _hash_seed(path: str, aug_id: int, salt: int = 12345) -> int:
    s = f"{path}::{aug_id}::{salt}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1)

def _aug_deterministic(seq_xyz: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = seq_xyz.copy()
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
    if rng.rand() < 0.8:
        s = rng.uniform(0.9, 1.1)
        out = out * s
    if rng.rand() < 0.5:
        noise = rng.normal(loc=0.0, scale=0.01, size=out.shape).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        out = out + noise
    return out

def _cache_file_path(cache_dir: Path, base_path: str, aug_id: int) -> Path:
    stem = hashlib.blake2b(base_path.encode("utf-8"), digest_size=8).hexdigest()
    return cache_dir / f"{stem}_aug{aug_id}.npy"

# -------------------------
# Frame & sequence loading (supports multiple per-frame layouts)
# -------------------------
def _load_frame_array(path: str) -> np.ndarray:
    """
    Load a single-frame .npy and return array shaped (V, 3) where V is number of joints.
    Handles common layouts:
      - (V,3) already -> return
      - (V,4) -> drop last column (visibility) and return (V,3)
      - flattened 1D arrays: attempts to detect mixed layouts where pose joints have 4 dims
         and hands have 3 dims (common with MediaPipe extraction).
         Specifically recognizes total length = 2*21*3 + 33*4 = 258 (hands + pose-with-vis).
         It handles either order: [hands(42*3) | pose(33*4)] or [pose(33*4) | hands(42*3)].
      - (1, V, C) -> squeeze to (V, C)
      - (V, C, 1) -> squeeze last dim
    Raises informative errors on unsupported shapes.
    """
    arr = np.load(path, allow_pickle=False)
    # If already (V, C) or (V, C, 1) or (1, V, C)
    if arr.ndim == 2:
        V, C = arr.shape
        if C == 3:
            return arr.astype(np.float32)
        if C == 4:
            # drop visibility
            return arr[:, :3].astype(np.float32)
        # ambiguous 2D: try to interpret as (V, C) anyway
        if C in (2,):  # likely not, but allow
            return arr.astype(np.float32)
        # Fallback: attempt to reshape if possible (rare)
        raise ValueError(f"Unsupported 2D frame shape {arr.shape} in {path}")

    if arr.ndim == 3:
        # common cases: (1, V, C) or (V, C, 1)
        if arr.shape[0] == 1:
            out = arr.squeeze(0)
            return _load_frame_array_from_ndarray(out, path)  # helper below
        if arr.shape[-1] == 1:
            out = arr[..., 0]
            return _load_frame_array_from_ndarray(out, path)
        # otherwise try to reduce to (V,C)
        if arr.shape[0] > 1 and arr.shape[1] in (2,3,4):
            # maybe (T, V, C) with T > 1 — but this loader expects single-frame; use first slice
            out = arr[0]
            return _load_frame_array_from_ndarray(out, path)
        raise ValueError(f"Unsupported 3D frame shape {arr.shape} in {path}")

    if arr.ndim == 1:
        X = arr.size
        # Heuristic: Mixed MediaPipe style:
        hands_cnt = 2 * 21  # left + right
        pose_cnt = 33
        if X == hands_cnt * 3 + pose_cnt * 4:
            # Two plausible orders: hands then pose, or pose then hands.
            # Try hands-first: first hands_cnt*3 values -> reshape to (42,3); tail -> (33,4)
            hands_len = hands_cnt * 3
            pose_len = pose_cnt * 4
            # try hands-first
            hands_block = arr[:hands_len].reshape((hands_cnt, 3)).astype(np.float32)
            pose_block = arr[hands_len:hands_len + pose_len].reshape((pose_cnt, 4)).astype(np.float32)
            # Validate reasonable ranges (optional): check finite
            if np.isfinite(hands_block).all() and np.isfinite(pose_block).all():
                # drop pose visibility column and concatenate: order -> [hands_left(21), hands_right(21), pose(33)]
                combined = np.vstack([hands_block, pose_block[:, :3]])
                if combined.shape[0] == 75 and combined.shape[1] == 3:
                    return combined
            # try pose-first
            pose_block = arr[:pose_len].reshape((pose_cnt, 4)).astype(np.float32)
            hands_block = arr[pose_len:pose_len + hands_len].reshape((hands_cnt, 3)).astype(np.float32)
            if np.isfinite(hands_block).all() and np.isfinite(pose_block).all():
                combined = np.vstack([hands_block, pose_block[:, :3]])
                if combined.shape[0] == 75 and combined.shape[1] == 3:
                    return combined
            # if both plausible but ordering differs from expected, return the one where pose visibility looks valid
            # fallback: return hands-first extraction
            return np.vstack([hands_block, pose_block[:, :3]]).astype(np.float32)

        # Generic flattened case: try to infer channels (prefer 4 detection for pose-heavy cases)
        # If divisible by 3 but also corresponds to a known mixed pattern, above handled it.
        # Here try simple heuristics:
        if X % 3 == 0:
            C = 3
            V = X // 3
            return arr.reshape((V, C)).astype(np.float32)
        if X % 4 == 0:
            C = 4
            V = X // 4
            return arr.reshape((V, C))[:, :3].astype(np.float32)  # drop last col
        # fallback: cannot infer
        raise ValueError(f"Cannot infer frame layout from flattened array length={X} in {path}")

    # unreachable in normal cases
    raise ValueError(f"Unsupported array ndim={arr.ndim} in {path}")

# helper used above for 3D cases to reuse same logic for 2D-like arrays
def _load_frame_array_from_ndarray(out: np.ndarray, path: str) -> np.ndarray:
    """
    out is expected to be 2D now (V,C) after squeezing or indexing.
    """
    if out.ndim != 2:
        raise ValueError(f"After squeeze expected 2D array, got shape {out.shape} from {path}")
    V, C = out.shape
    if C == 3:
        return out.astype(np.float32)
    if C == 4:
        return out[:, :3].astype(np.float32)
    # otherwise raise
    raise ValueError(f"Unsupported squeezed frame shape {(V,C)} in {path}")


def _load_sequence_from_folder(seq_folder: str) -> np.ndarray:
    """
    Read sorted .npy frames in seq_folder and stack -> [T, V, C].
    Sorting is numeric when filenames are like 0001.npy; otherwise lexicographic.
    """
    p = Path(seq_folder)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".npy"]
    if not files:
        raise FileNotFoundError(f"No .npy frames found in {seq_folder}")
    # try numeric sort based on stem
    try:
        files_sorted = sorted(files, key=lambda x: int(x.stem))
    except Exception:
        files_sorted = sorted(files, key=lambda x: x.name)
    frames = []
    for f in files_sorted:
        frame = _load_frame_array(str(f))  # (V,C)
        frames.append(frame)
    # ensure consistent V,C across frames
    V0, C0 = frames[0].shape
    for i,fr in enumerate(frames):
        if fr.shape != (V0, C0):
            # attempt to reshape flattened frame
            if fr.ndim == 2 and fr.shape[0] * fr.shape[1] == V0 * C0:
                frames[i] = fr.reshape((V0, C0))
            else:
                raise ValueError(f"Inconsistent frame shape in {seq_folder}: {f} -> {fr.shape} != {(V0,C0)}")
    stacked = np.stack(frames, axis=0).astype(np.float32)  # [T, V, C]
    return stacked

# -------------------------
# Build items by scanning root split directory
# -------------------------
def build_items_from_split(root_split_dir: str, seq_folder_level: int = 2) -> Tuple[List[Tuple[str,int]], List[str]]:
    """
    Scan root_split_dir expecting structure:
      root_split_dir/<action_label>/<sequence_folder>/frames.npy...
    Returns:
      - items: list of (sequence_folder_path, label_idx)
      - class_names: list mapping label_idx -> folder name (sorted)
    """
    root = Path(root_split_dir)
    if not root.exists():
        raise FileNotFoundError(root_split_dir)
    action_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    class_names = [p.name for p in action_dirs]
    label_map = {name: idx for idx, name in enumerate(class_names)}
    items: List[Tuple[str,int]] = []
    for action in action_dirs:
        for seq in sorted(action.iterdir()):
            if not seq.is_dir():
                continue
            items.append((str(seq.resolve()), label_map[action.name]))
    if not items:
        raise RuntimeError(f"No sequence folders found under {root_split_dir}")
    return items, class_names

# -------------------------
# Dataset
# -------------------------
class IndianSignSequenceDataset(Dataset):
    """
    Args:
      root_split_dir: root directory for the split (e.g., train path)
      max_T: target frames after temporal resampling
      feat: "xyz" or "xyz+vel"
      normalize: center on root_joint and scale by heuristic
      temporal_mode: "interp" or "crop_repeat"
      offline_aug_factor / offline_cache_dir / build_cache: as in IPN loader
    """
    def __init__(
        self,
        root_split_dir: str,
        max_T: int = 30,
        feat: str = "xyz",
        normalize: bool = True,
        root_joint: int = 0,
        temporal_mode: str = "interp",
        aug: bool = False,
        eval_mode: bool = False,
        dtype: str = "float32",
        offline_aug_factor: int = 0,
        offline_cache_dir: Optional[str] = None,
        build_cache: bool = False,
    ) -> None:
        super().__init__()
        self.root_split_dir = root_split_dir
        self.max_T = int(max_T)
        self.feat = feat.lower()
        self.normalize_flag = bool(normalize)
        self.root_joint = int(root_joint)
        self.temporal_mode = temporal_mode
        self.aug = bool(aug)
        self.eval_mode = bool(eval_mode)
        self.dtype = np.float32 if dtype == "float32" else np.float16

        self.offline_aug_factor = max(0, int(offline_aug_factor))
        self.cache_dir = Path(offline_cache_dir) if offline_cache_dir else None
        self.build_cache = bool(build_cache)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # build items by scanning folders
        self.items, self.class_names = build_items_from_split(root_split_dir)
        # example item: ("/abs/path/to/seq_folder", label)

        # expand items with offline augmentation IDs
        self.items_aug: List[Tuple[str,int,int]] = []
        K = self.offline_aug_factor if not self.eval_mode else 0
        for (p, y) in self.items:
            self.items_aug.append((p, y, 0))
            for k in range(1, K+1):
                self.items_aug.append((p, y, k))

        # inspect one sequence to set V,C
        sample_seq = _load_sequence_from_folder(self.items[0][0])
        self.V = sample_seq.shape[1]
        self.C = sample_seq.shape[2]
        self.M = 1

    def __len__(self) -> int:
        return len(self.items_aug)

    def _temporal_sample(self, seq_xyz: np.ndarray) -> np.ndarray:
        T = seq_xyz.shape[0]
        if self.max_T is None or self.max_T <= 0:
            return seq_xyz
        if self.temporal_mode == "interp":
            return _linear_time_resample(seq_xyz, self.max_T)
        # crop_repeat mode
        if T >= self.max_T:
            start = 0 if (not self.eval_mode or T == self.max_T) else np.random.randint(0, T-self.max_T+1)
            return seq_xyz[start:start+self.max_T]
        reps = int(np.ceil(self.max_T / T))
        idx = np.tile(np.arange(T, dtype=int), reps)[:self.max_T]
        if not self.eval_mode and self.max_T > T:
            shift = np.random.randint(0, T)
            idx = np.roll(idx, shift)
        return seq_xyz[idx]

    def _normalize_xyz(self, seq_xyz: np.ndarray) -> np.ndarray:
        if not self.normalize_flag:
            return seq_xyz
        try:
            seq = _root_center_skeleton(seq_xyz, self.root_joint)
        except Exception:
            seq = seq_xyz
        T, V, C = seq.shape
        if V >= 2 and C >= 3:
            diffs = seq[:, 1:, :3] - seq[:, :-1, :3]
            lens = np.sqrt((diffs**2).sum(-1) + 1e-6)
            mean_bone = max(float(lens.mean()), 1e-6)
            seq = seq / mean_bone
        return seq

    def _online_augment(self, seq_xyz: np.ndarray) -> np.ndarray:
        if self.eval_mode or not self.aug:
            return seq_xyz
        if np.random.rand() < 0.5:
            ang = np.deg2rad(np.random.uniform(-15,15))
            c, s = np.cos(ang), np.sin(ang)
            Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            seq_xyz[..., :3] = seq_xyz[..., :3] @ Rz.T
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            seq_xyz[..., :3] = seq_xyz[..., :3] * scale
        if np.random.rand() < 0.3:
            jitter = np.random.normal(0.0, 0.01, size=seq_xyz[..., :3].shape).astype(np.float32)
            seq_xyz[..., :3] += np.clip(jitter, -0.05, 0.05)
        return seq_xyz

    def _to_tensor_CTVM(self, seq_xyz: np.ndarray) -> torch.Tensor:
        if self.feat == "xyz+vel":
            xv = _add_velocity_channel(seq_xyz[..., :3])
            data = np.transpose(xv, (2, 0, 1))  # [C,T,V]
        elif self.feat == "xyz":
            data = np.transpose(seq_xyz[..., :3], (2, 0, 1))
        else:
            raise ValueError(f"Unsupported feat={self.feat}")
        data = data[..., np.newaxis]  # [C,T,V,1]
        return torch.from_numpy(data.astype(self.dtype))

    def _load_or_build_offline_aug(self, base_seq_folder: str, aug_id: int) -> np.ndarray:
        assert aug_id >= 1
        if self.cache_dir is None:
            seq = _load_sequence_from_folder(base_seq_folder)
            seq = self._temporal_sample(seq)
            seq = self._normalize_xyz(seq)
            seed = _hash_seed(base_seq_folder, aug_id)
            seq = _aug_deterministic(seq, seed)
            return seq.astype(self.dtype)
        cache_file = _cache_file_path(self.cache_dir, base_seq_folder, aug_id)
        if cache_file.exists():
            arr = np.load(str(cache_file))
            return arr.astype(self.dtype)
        if not self.build_cache:
            seq = _load_sequence_from_folder(base_seq_folder)
            seq = self._temporal_sample(seq)
            seq = self._normalize_xyz(seq)
            seed = _hash_seed(base_seq_folder, aug_id)
            seq = _aug_deterministic(seq, seed)
            return seq.astype(self.dtype)
        seq = _load_sequence_from_folder(base_seq_folder)
        seq = self._temporal_sample(seq)
        seq = self._normalize_xyz(seq)
        seed = _hash_seed(base_seq_folder, aug_id)
        seq = _aug_deterministic(seq, seed)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_file), seq.astype(np.float32))
        return seq.astype(self.dtype)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_seq_folder, label, aug_id = self.items_aug[idx]
        if aug_id == 0:
            seq = _load_sequence_from_folder(base_seq_folder)
            seq = self._temporal_sample(seq)
            seq = self._normalize_xyz(seq)
            if self.offline_aug_factor == 0:
                seq = self._online_augment(seq)
        else:
            seq = self._load_or_build_offline_aug(base_seq_folder, aug_id)
        data = self._to_tensor_CTVM(seq.astype(self.dtype))
        y = torch.tensor(label, dtype=torch.long)
        meta = {"seq_folder": base_seq_folder, "T": int(seq.shape[0]), "V": int(seq.shape[1]), "aug_id": int(aug_id)}
        return {"data": data, "label": y, "meta": meta}

    def print_classes(self) -> None:
        print("Number of classes:", len(self.class_names))
        for i, name in enumerate(self.class_names):
            print(f"{i}: {name}")

# -------------------------
# Collate & cache builder
# -------------------------
def collate_sign(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    datas  = torch.stack([b["data"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    metas  = [b["meta"] for b in batch]
    return {"data": datas, "label": labels, "meta": metas}

def build_offline_cache_for_split(root_split_dir: str, cache_dir: str, max_T: int = 30,
                                  feat: str = "xyz", offline_aug_factor: int = 2) -> None:
    ds = IndianSignSequenceDataset(
        root_split_dir=root_split_dir,
        max_T=max_T,
        feat=feat,
        aug=False,
        eval_mode=False,
        offline_aug_factor=offline_aug_factor,
        offline_cache_dir=cache_dir,
        build_cache=True,
    )
    for i in range(len(ds)):
        _ = ds[i]

# -------------------------
# Simple CLI for quick testing
# -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", help="root split dir, e.g. train folder", required=True)
    ap.add_argument("--max_T", type=int, default=30)
    ap.add_argument("--offline_aug_factor", type=int, default=0)
    ap.add_argument("--offline_cache_dir", type=str, default=None)
    ap.add_argument("--build_cache", action="store_true")
    args = ap.parse_args()

    ds = IndianSignSequenceDataset(
        root_split_dir=args.split_dir,
        max_T=args.max_T,
        feat="xyz",
        offline_aug_factor=args.offline_aug_factor,
        offline_cache_dir=args.offline_cache_dir,
        build_cache=args.build_cache,
    )
    print("Dataset size:", len(ds))
    ds.print_classes()
    ex = ds[0]
    print("Sample data shape:", ex["data"].shape, "label:", ex["label"].item(), "meta:", ex["meta"])
