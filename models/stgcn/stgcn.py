

# models/stgcn/stgcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tgcn import ConvTemporalGraphical
from utils.graph import Graph
from models.cfot import SignedAdaptiveDeltaCFOTv2, SignedMultiDeltaCFOT, CFOTLayer


# ---- Robust CFOT import (with clear diagnostics) ----
try:
    from models.cfot import (
        AdaptiveDeltaCFOTv2,
        MultiDeltaCFOT,
        SignedAdaptiveDeltaCFOTv2,   # new (handles negative deltas)
        SignedMultiDeltaCFOT,        # new (handles negative deltas)
    )
    _HAS_CFOT = True
    _CFOT_IMPORT_ERR = None
except Exception as _e:
    AdaptiveDeltaCFOTv2 = None
    MultiDeltaCFOT = None
    SignedAdaptiveDeltaCFOTv2 = None
    SignedMultiDeltaCFOT = None
    _HAS_CFOT = False
    _CFOT_IMPORT_ERR = repr(_e)


class Model(nn.Module):
    r"""ST-GCN with optional CFOT.

    Kwargs (if enable_cfot=True):
        cfot_type: "adaptive" | "multi"
        cfot_deltas: list[int] (can include negatives if Signed* is available)
        cfot_hidden: int
        cfot_iters: int
        cfot_tau: float
        cfot_topk: int
        cfot_beta: float
        cfot_inject: "pre" | "after1"
        cfot_sparsify / cfot_keep_mass / cfot_min_k / cfot_max_k (if supported by CFOT)
        temporal_kernel_size: odd int
    """
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()

        # ---- graph ----
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # ---- parse known kwargs up front (POP) ----
        temporal_kernel_size = int(kwargs.pop("temporal_kernel_size", 9))
        assert temporal_kernel_size % 2 == 1, "temporal_kernel_size must be odd"
        dropout = float(kwargs.pop("dropout", 0.0))

        # CFOT args
        self.cfot_module = None
        self.cfot_beta = float(kwargs.pop("cfot_beta", 1.0))
        self.cfot_strength_frac = 0.0
        enable_cfot = bool(kwargs.pop("enable_cfot", False))
        cfot_type   = str(kwargs.pop("cfot_type", "adaptive"))
        deltas      = kwargs.pop("cfot_deltas", [1])
        emb         = int(kwargs.pop("cfot_hidden", 64))
        iters       = int(kwargs.pop("cfot_iters", 9))
        tau         = float(kwargs.pop("cfot_tau", 0.45))
        topk        = int(kwargs.pop("cfot_topk", 3))
        self.cfot_inject = str(kwargs.pop("cfot_inject", "after1")).lower()
        assert self.cfot_inject in {"pre", "after1"}
        cfot_in_ch = in_channels if self.cfot_inject == "pre" else 64

        cfot_affinity     = str(kwargs.pop("cfot_affinity", "euclid")).lower()
        cfot_euclid_scale = float(kwargs.pop("cfot_euclid_scale", 1.0))
        cfot_cosine_eps   = float(kwargs.pop("cfot_cosine_eps", 1e-6))
        cfot_pos_weight   = float(kwargs.pop("cfot_pos_weight", 1.0))
        cfot_vel_weight   = float(kwargs.pop("cfot_vel_weight", 0.2))


        # optional sparsity knobs (passed through only if CFOT supports them)
        self.cfot_sparsify   = kwargs.pop("cfot_sparsify", "topk")
        self.cfot_keep_mass  = float(kwargs.pop("cfot_keep_mass", 0.4))
        self.cfot_min_k      = int(kwargs.pop("cfot_min_k", 2))
        cfot_max_k_raw       = kwargs.pop("cfot_max_k", None)
        self.cfot_max_k      = int(cfot_max_k_raw) if (cfot_max_k_raw is not None) else None

        # ---- ST-GCN backbone ----
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64,  kernel_size, 1, residual=False, dropout=dropout),
            st_gcn(64,         64,  kernel_size, 1, dropout=dropout),
            st_gcn(64,         64,  kernel_size, 1, dropout=dropout),
            st_gcn(64,         64,  kernel_size, 1, dropout=dropout),
            st_gcn(64,         128, kernel_size, 2, dropout=dropout),
            st_gcn(128,        128, kernel_size, 1, dropout=dropout),
            st_gcn(128,        128, kernel_size, 1, dropout=dropout),
            st_gcn(128,        256, kernel_size, 2, dropout=dropout),
            st_gcn(256,        256, kernel_size, 1, dropout=dropout),
            st_gcn(256,        256, kernel_size, 1, dropout=dropout),
        ))

        # edge importance
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # classifier head
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        # ---- CFOT (optional) ----
        if enable_cfot:
            if not _HAS_CFOT:
                raise RuntimeError(f"CFOT requested but import failed: {_CFOT_IMPORT_ERR}")

            use_signed = any(int(d) < 0 for d in deltas)
            if cfot_type == "adaptive":
                CFOTCls = SignedAdaptiveDeltaCFOTv2 if use_signed and (SignedAdaptiveDeltaCFOTv2 is not None) else AdaptiveDeltaCFOTv2
            else:
                CFOTCls = SignedMultiDeltaCFOT if use_signed and (SignedMultiDeltaCFOT is not None) else MultiDeltaCFOT

            # Build CFOT; pass only widely-supported args
            self.cfot_module = CFOTCls(
                in_channels=cfot_in_ch,
                emb_channels=emb,
                deltas=deltas,
                sinkhorn_iters=iters,
                sinkhorn_tau=tau,
                topk=topk,
                store_links=True,
                affinity=cfot_affinity,
                euclid_scale=cfot_euclid_scale,
                cosine_eps=cfot_cosine_eps,
                pos_weight=cfot_pos_weight,
                vel_weight=cfot_vel_weight,
                cfot_sparsify=self.cfot_sparsify,
                cfot_keep_mass=self.cfot_keep_mass,
                cfot_min_k=self.cfot_min_k,
                cfot_max_k=(self.cfot_max_k if self.cfot_max_k is not None else V),
            )
            print(f"[stgcn] CFOT attached: type={cfot_type}{' (signed)' if use_signed else ''} "
                    f"deltas={deltas} emb={emb} iters={iters} tau={tau} topk={topk} affinity={cfot_affinity}")


        # ---- Viz placeholders for trainer/logger ----
        self.cfot_last_adj_by_delta = None   # (D,V,V) or None
        self.cfot_last_avg_adj      = None   # (V,V) or None
        self.cfot_deltas            = list(deltas) if isinstance(deltas, (list, tuple)) else [1]

    # allow scheduler to ramp strength
    def set_cfot_strength(self, frac: float):
        self.cfot_strength_frac = float(max(0.0, min(1.0, frac)))

    # ---- small helpers to keep CFOT handling consistent ----
    def _stash_cfot_links(self):
        if self.cfot_module is None:
            self.cfot_last_adj_by_delta = None
            self.cfot_last_avg_adj = None
            return
        try:
            A_by_d = getattr(self.cfot_module, "last_adj_by_delta", None)  # [D,V,V]
            if isinstance(A_by_d, torch.Tensor):
                self.cfot_last_adj_by_delta = A_by_d.detach().cpu()
                self.cfot_last_avg_adj = A_by_d.mean(dim=0).detach().cpu()
        except Exception:
            pass

    def _apply_cfot(self, x):
        """Run CFOT (always refresh links); add residual only if strength>0."""
        if self.cfot_module is None:
            return x
        r = self.cfot_module(x)
        self._stash_cfot_links()
        if self.cfot_strength_frac <= 0.0 or self.cfot_beta == 0.0:
            return x
        return x + (self.cfot_beta * self.cfot_strength_frac) * r

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # ---- CFOT injection "pre" (before any GCN block) ----
        cfot_applied = False
        if self.cfot_inject == "pre" and not cfot_applied:
            x = self._apply_cfot(x)
            cfot_applied = True

        for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _ = gcn(x, self.A * importance)
            if (i == 0) and (self.cfot_inject == "after1") and not cfot_applied:
                x = self._apply_cfot(x)
                cfot_applied = True

        # ---- head ----
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)

    def extract_feature(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        if self.cfot_inject == "pre":
            x = self._apply_cfot(x)

        for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _ = gcn(x, self.A * importance)
            if (i == 0) and (self.cfot_inject == "after1"):
                x = self._apply_cfot(x)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        out = self.fcn(x)
        output = out.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        return output, feature


class st_gcn(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True, **kwargs):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        pad_t = (kernel_size[0] - 1) // 2
        padding = (pad_t, 0)

        # spatial GCN over A_k’s (K=kernel_size[1])
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # temporal conv + norm + drop
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # --- residual path (MUST define the attribute) ---
        if not residual:
            # zero mapping
            class _Zero(nn.Module):
                def forward(self, x): return 0
            self.residual = _Zero()
        elif (in_channels == out_channels) and (stride == 1):
            # identity
            self.residual = nn.Identity()
        else:
            # match channels / stride
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

        # --- CFOT hooks (safe defaults; used by your forward) ---
        self.cfot_module = kwargs.get("cfot_module", None)
        self.cfot_inject = kwargs.get("cfot_inject", "pre")  # "pre" or "post"
        self._beta_runtime = float(kwargs.get("cfot_beta", 1.0))


    # def forward(self, x, A):
        # res = self.residual(x)
        # x, A = self.gcn(x, A)
        # x = self.tcn(x) + res
        # return self.relu(x), A


    def forward(self, x, A=None):
        # x: [B, C, T, V]
        # Residual branch from the original input (before TCN)
        res = self.residual(x)

        # Optional CFOT injection (pre-GCN)
        x_in = x
        if hasattr(self, "cfot_module") and (self.cfot_module is not None) and getattr(self, "cfot_inject", "pre") == "pre":
            beta = float(getattr(self, "_beta_runtime", 1.0))
            x_in = x + beta * self.cfot_module(x)   # same shape as x

        # Graph convolution (spatial) + temporal conv stack
        # ConvTemporalGraphical typically returns (y, A_out). If yours returns only y, just set A_out = A.
        out = self.gcn(x_in, A)
        if isinstance(out, tuple):
            y, A_out = out
        else:
            y, A_out = out, A

        y = self.tcn(y)           # temporal conv
        y = y + res               # add residual
        y = self.relu(y)          # nonlinearity

        # Optional CFOT injection (post-GCN/TCN)
        if hasattr(self, "cfot_module") and (self.cfot_module is not None) and getattr(self, "cfot_inject", "pre") == "post":
            beta = float(getattr(self, "_beta_runtime", 1.0))
            y = y + beta * self.cfot_module(y)

        return y, A_out


