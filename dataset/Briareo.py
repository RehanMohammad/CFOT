# briareo.py
"""
Briareo Hand Gesture Dataset Loader (21 joints)
-----------------------------------------------

Output: dict with
  data:  [C, T, V, 1]   V=21
  label: LongTensor
  meta:  {'path': str, 'T': int, 'V': 21}

Options (mirrors IPN loader):
  - max_T, temporal_mode {"crop_repeat","interp"}
  - normalize (root-center + unit-bone scale)
  - aug (train only), eval_mode
  - label_map optional reindexing
"""

from __future__ import annotations
import os
import re
import glob
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

# ------------------------------
# Try shared utils; fall back locally
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

# MediaPipe-like hand edges for scaling
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
    vel = np.zeros_like(seq_xyz)
    vel[1:] = seq_xyz[1:] - seq_xyz[:-1]
    return np.concatenate([seq_xyz, vel], axis=2)  # [T,V,6]

# ------------------------------
# Optional augmentation helpers
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
# Robust NPZ/NPY split reader
# ------------------------------
def _find_seq_txt(seq_hint: str, data_root: Optional[str]) -> Optional[str]:
    """Resolve a sequence hint into a .txt file."""
    h = str(seq_hint).replace("\\", "/")
    # direct file
    if h.lower().endswith(".txt") and os.path.isfile(h):
        return h
    # try under data_root
    if data_root:
        p = os.path.join(data_root, h)
        if p.lower().endswith(".txt") and os.path.isfile(p):
            return p
        if os.path.isdir(p):
            cands = sorted(glob.glob(os.path.join(p, "*.txt")))
            if cands:
                return cands[0]
        # folder inferred from hint
        parts = h.split("/")
        if len(parts) >= 2:
            folder = os.path.join(data_root, *parts[:-1])
            if os.path.isdir(folder):
                cands = sorted(glob.glob(os.path.join(folder, "*.txt")))
                if cands:
                    return cands[0]
    # if hint itself is a dir
    if os.path.isdir(h):
        cands = sorted(glob.glob(os.path.join(h, "*.txt")))
        if cands:
            return cands[0]
    # shallow recursive scan under data_root
    if data_root and os.path.isdir(data_root):
        base = data_root
        prefixes = [os.path.basename(h).split(".")[0]]
        for d in [base] + [os.path.join(base, x) for x in os.listdir(base)]:
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True):
                    name = os.path.basename(f)
                    if any(pref and pref in name for pref in prefixes):
                        return f
    return None

def _load_briareo_index(npy_path: str, data_root: Optional[str]) -> List[Tuple[str,int]]:
    """
    Accepts:
      - .npz with keys holding arrays/lists of dicts or tuples
      - .npy with object arrays or a dict-of-lists

    Each record must yield {'data': <path or list/array of paths>, 'label': int}.
    """
    items: List[Tuple[str,int]] = []

    def _yield_records_from(obj):
        # dict-like
        if hasattr(obj, "values") and callable(getattr(obj, "values")):
            for v in obj.values():
                if isinstance(v, (list, tuple)):
                    for e in v:
                        yield e
                else:
                    yield v
            return
        # numpy array containers
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                for e in obj.reshape(-1).tolist():
                    yield e
            else:
                for e in obj.reshape(-1):
                    yield e
            return
        # single dict or tuple
        yield obj

    # load container(s)
    if npy_path.lower().endswith(".npz"):
        z = np.load(npy_path, allow_pickle=True)
        containers = [z[k] for k in z.files]  # handle all arrays in the zip
    else:
        containers = [np.load(npy_path, allow_pickle=True)]

    # iterate and extract paths
    for cont in containers:
        for rec in _yield_records_from(cont):
            if not isinstance(rec, dict):
                continue
            if "label" not in rec or "data" not in rec:
                continue
            try:
                label = int(rec["label"])
            except Exception:
                continue

            seq_field = rec["data"]
            if isinstance(seq_field, (list, tuple, np.ndarray)) and len(seq_field) > 0:
                cand = str(seq_field[0])
            else:
                cand = str(seq_field)

            # legacy behavior: drop last two parts to get folder
            path = None
            parts = cand.replace("\\", "/").split("/")
            if len(parts) >= 2:
                folder_rel = "/".join(parts[:-2])
                folder_abs = os.path.join(data_root or "", folder_rel)
                if os.path.isdir(folder_abs):
                    cands = sorted(glob.glob(os.path.join(folder_abs, "*.txt")))
                    if cands:
                        path = cands[0]
            if path is None:
                path = _find_seq_txt(cand, data_root)

            if path and os.path.isfile(path):
                items.append((path, label))

    if not items:
        print(f"[Briareo] No items parsed from {npy_path}. "
              f"Verify NPZ keys and data_dir={data_root} root.")
    return items

# ------------------------------
# Robust TXT frame loader
# ------------------------------
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def _parse_joint_line(line: str) -> Optional[List[float]]:
    s = line.replace(";", " ").replace(",", " ").strip()
    if not s:
        return None
    nums = [float(m.group(0)) for m in _FLOAT_RE.finditer(s)]
    if len(nums) < 3:
        return None
    # some dumps prepend a non-xyz value (e.g., "degree/conn") → drop extras
    if len(nums) >= 4:
        nums = nums[-3:]
    x, y, z = nums[:3]
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        return None
    return [x, y, z]

def _finalize_frame(buf: List[List[float]], out_frames: List[np.ndarray]) -> None:
    if len(buf) == 0:
        return
    if len(buf) < 21:
        buf = buf + [[-1.0, -1.0, -1.0]] * (21 - len(buf))
    else:
        buf = buf[:21]
    arr = np.asarray(buf, dtype=np.float32)
    if np.isfinite(arr).sum() >= 3:
        out_frames.append(arr)

def _load_briareo_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    frames: List[np.ndarray] = []

    # mode A: blank-line separated frames
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    if len(blocks) >= 2:
        for blk in blocks:
            buf: List[List[float]] = []
            for ln in blk.splitlines():
                xyz = _parse_joint_line(ln)
                if xyz is not None:
                    buf.append(xyz)
            _finalize_frame(buf, frames)

    # mode B: stream all numeric lines and chunk by 21
    if len(frames) == 0:
        nums_lines: List[List[float]] = []
        for ln in raw.splitlines():
            xyz = _parse_joint_line(ln)
            if xyz is not None:
                nums_lines.append(xyz)
        for i in range(0, len(nums_lines), 21):
            _finalize_frame(nums_lines[i:i+21], frames)

    if len(frames) == 0:
        raise ValueError(f"No valid frames in {path}")
    if len(frames) == 1:
        frames.append(frames[0].copy())
    return np.stack(frames, axis=0)  # [T,21,3]

# ------------------------------
# Dataset
# ------------------------------
class BriareoDataset(Dataset):
    """
    Briareo → [C,T,V,1] with V=21
    """
    _bad_reported = 0
    _bad_report_cap = 10

    def __init__(
        self,
        ann_file: str,
        data_dir: Optional[str] = None,
        max_T: int = 80,
        normalize: bool = True,
        feat: str = "xyz",               # {"xyz","xyz+vel"}
        root_joint: int = 0,
        temporal_mode: str = "interp",   # {"crop_repeat","interp"}
        aug: bool = False,
        eval_mode: bool = False,
        label_map: Optional[Dict[int,int]] = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.ann_file = ann_file
        self.data_dir = data_dir
        self.max_T = int(max_T)
        self.normalize_flag = bool(normalize)
        self.feat = str(feat).lower()
        self.root_joint = int(root_joint)
        self.temporal_mode = str(temporal_mode)
        self.aug = bool(aug)
        self.eval_mode = bool(eval_mode)
        self.dtype = np.float32 if dtype == "float32" else np.float16

        raw_items = _load_briareo_index(ann_file, data_dir)   # [(txt_path,label)]
        self.items: List[Tuple[str,int]] = []
        for p, y in raw_items:
            y_new = label_map[y] if (label_map is not None and y in label_map) else y
            try:
                _ = _load_briareo_txt(p)   # filter broken upfront
                self.items.append((p, y_new))
            except Exception:
                # skip silently to keep parity with IPN loader behavior
                continue
        if not self.items:
            raise RuntimeError("No valid Briareo samples after filtering.")

        self.num_classes = (len(label_map) if label_map is not None
                            else (max(y for _,y in self.items)+1))
        self.V = 21
        self.M = 1

    def __len__(self) -> int:
        return len(self.items)

    # ---- preprocessing ----
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

    def _augment(self, seq_xyz: np.ndarray) -> np.ndarray:
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
        return torch.from_numpy(data.astype(self.dtype, copy=False))

    # ---- I/O ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tries = 0
        while tries < 5:
            path, label = self.items[idx]
            try:
                seq = _load_briareo_txt(path)               # [T,21,3]
                seq = self._temporal_sample(seq)
                seq = self._normalize_xyz(seq)
                seq = self._augment(seq).astype(self.dtype, copy=False)
                data = self._to_tensor_CTVM(seq)
                y = torch.tensor(label, dtype=torch.long)
                meta = {"path": path, "T": int(seq.shape[0]), "V": 21}
                return {"data": data, "label": y, "meta": meta}
            except Exception as e:
                if BriareoDataset._bad_reported < BriareoDataset._bad_report_cap:
                    print(f"[Briareo] malformed file: {path} ({e})")
                    BriareoDataset._bad_reported += 1
                idx = np.random.randint(0, len(self.items))
                tries += 1
        raise RuntimeError("Too many malformed Briareo samples encountered.")

# ------------------------------
# Collate
# ------------------------------
def briareo_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    datas  = torch.stack([b["data"] for b in batch], dim=0)  # [N,C,T,V,1]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    metas  = [b["meta"] for b in batch]
    return {"data": datas, "label": labels, "meta": metas}

# ------------------------------
# Label map builder (optional)
# ------------------------------
def build_briareo_label_map(ann_npy: str, data_dir: Optional[str]) -> Dict[int,int]:
    items = _load_briareo_index(ann_npy, data_dir)
    raw = sorted({y for _, y in items})
    return {y:i for i,y in enumerate(raw)}

# ------------------------------
# Self-test
# ------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Briareo annotations .npz/.npy")
    ap.add_argument("--data_dir", required=True, help="Root folder for sequences or .txt files")
    ap.add_argument("--max_T", type=int, default=80)
    ap.add_argument("--temporal_mode", default="interp", choices=["crop_repeat","interp"])
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--feat", default="xyz", choices=["xyz","xyz+vel"])
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    ds = BriareoDataset(
        ann_file=args.ann,
        data_dir=args.data_dir,
        max_T=args.max_T,
        temporal_mode=args.temporal_mode,
        normalize=not args.no_normalize,
        feat=args.feat,
        aug=not args.no_aug,
        eval_mode=bool(args.eval),
    )
    print(f"Dataset size: {len(ds)}  classes: {ds.num_classes}")
    ex = ds[0]
    print("Sample:", ex["data"].shape, "label:", ex["label"].item(), "meta:", ex["meta"])
