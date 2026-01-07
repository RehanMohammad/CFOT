# -----------------------------------------
# Cross-Frame Optimal Transport (CFOT) ops
# -----------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Sinkhorn", "CFOTLayer", "AdaptiveDeltaCFOTv2", "MultiDeltaCFOT"]


# -----------------------------
# Sinkhorn with numerical guards
# -----------------------------
class Sinkhorn(nn.Module):
    def __init__(self, iters: int = 5, eps: float = 1e-6, clamp: float = 8.0):
        super().__init__()
        self.iters = int(iters)
        self.eps = float(eps)
        self.clamp = float(clamp)

    def forward(self, scores: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """
        scores: [B, T', V, V]  -> approx doubly-stochastic [B, T', V, V]
        """
        tau = max(float(tau), 1e-3)
        S = torch.clamp(scores, -self.clamp, self.clamp)
        K = torch.exp(S / tau)
        K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0) + self.eps

        for _ in range(self.iters):
            K = K / (K.sum(dim=-1, keepdim=True) + self.eps)  # row
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0) + self.eps
            K = K / (K.sum(dim=-2, keepdim=True) + self.eps)  # col
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0) + self.eps

        return torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)



# ----------------------------------------
# Cross-Frame Optimal Transport (single Δ)
# ----------------------------------------
class CFOTLayer(nn.Module):
    """
    Cross-Frame Optimal Transport over joints with stride Δ (Δ >= 1).
    Adds L2 velocity-magnitude term to the affinity.

    Input:  x  [B, C, T, V]
    Output: R  [B, C, T, V]  (zeros for t < Δ; messages transported from t -> t+Δ)
    """
    def __init__(self,
        in_channels: int,
        emb_channels: int = 64,
        delta: int = 1,
        sinkhorn_iters: int = 5,
        sinkhorn_tau: float = 0.5,
        topk: int = 4,
        pos_weight: float = 1.0,
        vel_weight: float = 0.2,
        beta: float = 1.0,
        eps: float = 1e-6,
        clamp: float = 8.0,
        cfot_sparsify: str = "adaptive",
        keep_mass: float = 0.90,
        min_k: int = 2,
        max_k: int | None = None,
        affinity: str = "euclid",          # "euclid" | "cosine" | "learned"
        euclid_scale: float = 1.0,         # optional scale for -||a-b||2
        cosine_eps: float = 1e-6,          # numerical guard for cosine
        transport: str = "sinkhorn",
    ):
        super().__init__()
        assert delta >= 1, "This layer assumes causal Δ>=1."
        self.delta = int(delta)
        self.topk = int(topk)
        self.tau = float(sinkhorn_tau)
        self.eps = float(eps)
        self.clamp = float(clamp)
        self.pos_w = float(pos_weight)
        self.vel_w = float(vel_weight)
        self.affinity = str(affinity).lower()
        self.euclid_scale = float(euclid_scale)
        self.cosine_eps = float(cosine_eps)
        self.transport = transport.lower()
        assert self.transport in ("sinkhorn", "softmax")


        # Embeddings
        self.embed = nn.Conv2d(in_channels, emb_channels, kernel_size=1, bias=False)
        self.bn_e  = nn.BatchNorm2d(emb_channels)
        self.f1 = nn.Conv2d(emb_channels, emb_channels, kernel_size=1, bias=False)
        self.f2 = nn.Conv2d(emb_channels, emb_channels, kernel_size=1, bias=False)

        # Projection before transport
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn_p = nn.BatchNorm2d(in_channels)


        self.sk = Sinkhorn(iters=sinkhorn_iters, eps=eps, clamp=clamp)

        # Init
        nn.init.kaiming_uniform_(self.embed.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.f1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.f2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

        # Debug/inspection
        self.store_links = False
        self.last_links = None

        self._beta_runtime = float(beta)  # kept for symmetry with older APIs

        self.cfot_sparsify = str(cfot_sparsify)
        self.keep_mass = float(keep_mass)
        self.min_k = int(min_k)
        self.max_k = (int(max_k) if max_k is not None else None)
        self.cfot_keep_mass = self.keep_mass
        self.cfot_min_k = self.min_k
        self.cfot_max_k = self.max_k

    @torch.no_grad()
    def set_sparsify_mode(self, mode: str = "topk", keep_mass: float = 0.9,
                          min_k: int = 2, max_k: int | None = None):
        self.sparsify = str(mode); self.cfot_sparsify = self.sparsify
        self.keep_mass = float(keep_mass); self.cfot_keep_mass = self.keep_mass
        self.min_k = int(min_k); self.cfot_min_k = self.min_k
        self.max_k = (int(max_k) if max_k is not None else None)
        self.cfot_max_k = self.max_k

    def _sparsify(self, P: torch.Tensor) -> torch.Tensor:
        if P is None:
            raise RuntimeError(
                "CFOTLayer._sparsify received P=None. "
                "This usually means Sinkhorn.forward did not return a tensor. "
                "Check that Sinkhorn.forward ends with `return K`."
            )

        mode = getattr(self, "sparsify", getattr(self, "cfot_sparsify", "topk"))
        if mode == "topk":
            return self._sparsify_topk(P, self.topk)

        # adaptive “keep-mass”
        keep = float(getattr(self, "keep_mass", getattr(self, "cfot_keep_mass", 0.9)))
        min_k = int(getattr(self, "min_k", getattr(self, "cfot_min_k", 2)))
        max_k = getattr(self, "max_k", getattr(self, "cfot_max_k", None))

        B, T, Vsrc, Vtgt = P.shape
        vals, idx = torch.sort(P, dim=-2, descending=True)
        csum = torch.cumsum(vals, dim=-2)

        keep_mask = (csum < keep)
        kmin_mask = torch.zeros_like(keep_mask)
        if min_k > 0:
            kmin_mask[..., :min_k, :] = 1
        mask = keep_mask | kmin_mask.bool()

        if max_k is not None:
            kcap = torch.zeros_like(mask)
            kcap[..., :max_k, :] = 1
            mask = mask & kcap.bool()

        M = torch.zeros_like(P)
        M.scatter_(-2, idx, mask.float())
        P = P * M
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)
        return torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)


    @torch.no_grad()
    def set_beta(self, b: float):
        self._beta_runtime = float(b)

    def _pair_scores(
        self,
        e_t: torch.Tensor,
        e_td: torch.Tensor,
        v_t: torch.Tensor,
        v_td: torch.Tensor,
    ) -> torch.Tensor:
        """
        e_t, e_td: [B, D, T', V]
        v_t, v_td: [B, T', V]
        return: S [B, T', V, V] (higher is better)
        """
        B, D, Tp, V = e_t.shape

        # project once
        a_proj = self.f1(e_t)            # [B,D,T',V]
        b_proj = self.f2(e_td)           # [B,D,T',V]

        if self.affinity == "learned":
            # bilinear dot-product (current behavior)
            a = a_proj.permute(0, 2, 3, 1).contiguous().view(B * Tp, V, D)
            b = b_proj.permute(0, 2, 1, 3).contiguous().view(B * Tp, D, V)
            S_pos = torch.bmm(a, b).view(B, Tp, V, V) / math.sqrt(D)

        elif self.affinity == "cosine":
            # cosine over projected embeddings
            a = a_proj.permute(0, 2, 3, 1).contiguous().view(B * Tp, V, D)
            b = b_proj.permute(0, 2, 3, 1).contiguous().view(B * Tp, V, D)
            a = a / (a.norm(p=2, dim=-1, keepdim=True) + self.cosine_eps)
            b = b / (b.norm(p=2, dim=-1, keepdim=True) + self.cosine_eps)
            # (V,D)@(D,V) per time → (V,V)
            S_pos = torch.bmm(a, b.transpose(1, 2)).view(B, Tp, V, V)

        else:  # "euclid"
            # negative L2 distance between projected embeddings
            a = a_proj.permute(0, 2, 3, 1).contiguous().view(B * Tp, V, D)
            b = b_proj.permute(0, 2, 3, 1).contiguous().view(B * Tp, V, D)
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b, but compute robustly
            aa = (a * a).sum(-1, keepdim=True)          # [BTp,V,1]
            bb = (b * b).sum(-1, keepdim=True).transpose(1, 2)  # [BTp,1,V]
            ab = torch.bmm(a, b.transpose(1, 2))        # [BTp,V,V]
            dist2 = (aa + bb - 2.0 * ab).clamp_min(0.0)
            S_pos = (-torch.sqrt(dist2 + 1e-8) * self.euclid_scale).view(B, Tp, V, V)

        # velocity mismatch penalty
        Dv = (v_t.unsqueeze(-1) - v_td.unsqueeze(-2)).abs()
        S = self.pos_w * S_pos - self.vel_w * Dv
        S = torch.clamp(S, -self.clamp, self.clamp)
        return torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)


    def _sparsify_topk(self, P: torch.Tensor, k: int) -> torch.Tensor:
        """
        Keep top-k over src (row-wise). P: [B, T', V_src, V_tgt]
        """
        if k <= 0:
            return P
        Vsrc = P.size(-2)
        k = int(min(k, Vsrc))
        if k >= Vsrc:
            return P
        _, idx = torch.topk(P, k=k, dim=-2)  # idx: [B, T', k, V_tgt]
        mask = torch.zeros_like(P)
        mask.scatter_(-2, idx, 1.0)
        P = P * mask
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)
        return torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

    def _sparsify_adaptive(self, P: torch.Tensor, keep_mass: float = 0.9,
                       min_k: int = 2, max_k: int | None = None) -> torch.Tensor:
        """
        Adaptive per-(B,tgt_joint,t) sparsification.
        Keep the smallest set of source joints whose probs sum to >= keep_mass.
        P: [B, T', V_src, V_tgt] (rows=src, cols=tgt)
        """
        keep_mass = float(max(0.0, min(1.0, keep_mass)))
        if keep_mass <= 0.0:
            return torch.zeros_like(P)
        B, Tp, Vsrc, Vtgt = P.shape
        if max_k is None: max_k = Vsrc

        # sort rows (src) per column (tgt)
        vals, idx = torch.sort(P, dim=-2, descending=True)             # [B,T',Vsrc,Vtgt]
        cum = torch.cumsum(vals, dim=-2)                               # cumulative mass
        # how many rows needed to reach keep_mass (per B,T',Vtgt)
        keep_mask = (cum < keep_mass)                                  # True until we pass α
        k = keep_mask.sum(dim=-2) + 1 
        k = k.clamp(min=min_k, max=max_k)                              # [B,T',Vtgt]

        # build boolean mask in sorted space: rank < k
        ranks = torch.arange(Vsrc, device=P.device).view(1,1,Vsrc,1)   # [1,1,Vsrc,1]
        sorted_keep = (ranks < k.unsqueeze(-2))                        # [B,T',Vsrc,Vtgt]


        # scatter back to original row order
        mask = torch.zeros_like(P, dtype=P.dtype)
        mask.scatter_(-2, idx, sorted_keep.to(P.dtype))

        # apply mask and renormalize rows (row-stochastic by Sinkhorn → preserve that)
        P = P * mask
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)
        return torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)


    @torch.no_grad()
    def enable_store_links(self, flag: bool = True):
        self.store_links = bool(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, V] -> residual R: [B, C, T, V]
        C expects xyz in first 3 channels if available. Causal Δ>=1.
        """
        B, C, T, V = x.shape
        d = self.delta
        if T <= d:
            return torch.zeros_like(x)

        # Time windows
        Xt  = x[:, :, :-d, :]   # [B,C,T',V]
        Xtd = x[:, :,  d:, :]   # [B,C,T',V]

        # Per-frame velocity magnitude with length T, then align to T'
        if C >= 3:
            Xpos   = x[:, :3, :, :]                              # [B,3,T,V]
            dv     = Xpos[:, :, 1:, :] - Xpos[:, :, :-1, :]      # [B,3,T-1,V]
            v_step = torch.linalg.norm(dv, dim=1)                # [B,T-1,V]
            v0     = v_step.new_zeros(B, 1, V)                   # [B,1,V]
            v_frame = torch.cat([v0, v_step], dim=1)             # [B,T,V]
            v_t   = v_frame[:, :-d, :]                           # [B,T',V]
            v_td  = v_frame[:,  d:, :]                           # [B,T',V]
        else:
            v_t  = Xt.new_zeros(B, T - d, V)
            v_td = Xtd.new_zeros(B, T - d, V)

        # Embeddings
        e_t  = self.bn_e((self.embed(Xt)))                         # [B,D,T',V]
        e_td = self.bn_e((self.embed(Xtd)))                       # [B,D,T',V]

        # Sanity
        assert e_t.size(2) == v_t.size(1) == v_td.size(1), \
            f"T' mismatch: emb={e_t.size(2)} v_t={v_t.size(1)} v_td={v_td.size(1)}"

        # Affinity -> Sinkhorn -> sparsify
        S = self._pair_scores(e_t, e_td, v_t, v_td)              # [B,T',V,V]
        # P = self.sk(S, tau=self.tau)                             # [B,T',V,V]
        if self.transport == "sinkhorn":
            # Doubly-stochastic OT (original CFOT)
            P = self.sk(S, tau=self.tau)

        elif self.transport == "softmax":
            # Row-stochastic attention (ablation)
            P = torch.softmax(S / self.tau, dim=-2)

        else:
            raise ValueError(f"Unknown transport mode: {self.transport}")

        # P = self._sparsify_topk(P, self.topk)                    # [B,T',V,V]
        # P = self._sparsify_adaptive(P, keep_mass=0.9, min_k=2, max_k=V)

        # mode = getattr(self, "cfot_sparsify", getattr(self, "cfot_sparsify", "topk"))
        mode = getattr(self, "sparsify", getattr(self, "cfot_sparsify", "topk"))

        if mode == "adaptive":
            P = self._sparsify_adaptive(
                P,
                keep_mass=float(getattr(self, "keep_mass", getattr(self, "cfot_keep_mass", 0.9))),
                min_k=int(getattr(self, "min_k", getattr(self, "cfot_min_k", 2))),
                max_k=getattr(self, "max_k", getattr(self, "cfot_max_k", None)),
            )
        else:
            P = self._sparsify(P)  # calls topk path if mode=="topk"


        # Project + transport t -> t+Δ
        Yt  = self.bn_p(self.proj(Xt))                           # [B,C,T',V]
        msg = torch.einsum("btij,bcti->bctj", P, Yt)             # [B,C,T',V]
        msg = torch.nan_to_num(msg, nan=0.0, posinf=0.0, neginf=0.0)

        # Residual placement (zeros for first d frames)
        R = torch.zeros_like(x)
        R[:, :, d:, :] =  msg

        if self.store_links:
            # self.last_links = {"P": P.detach().cpu(), "delta": self.delta}
            # self.last_links = { "P": P,"R_fused": R,"delta": self.delta
            self.last_links = {
                        "delta": self.delta,
                        "S": S.detach(),
                        "P": P.detach() if P is not None else None
    }


        else:
            self.last_links = None

        return R



# ------------------------------------------------------------
# Adaptive-Δ CFOT v2: learned gate over multiple deltas
# ------------------------------------------------------------
class AdaptiveDeltaCFOTv2(nn.Module):
    """
    Learn to mix several CFOT deltas Δ∈{d1,d2,...} with a small temporal gating net.
    """
    def __init__(self,
        in_channels: int,
        emb_channels: int = 64,
        deltas=(1, 2, 4),
        sinkhorn_iters: int = 5,
        sinkhorn_tau: float = 0.5,
        topk: int = 4,
        gate_hidden: int = 32,
        hard_gating: bool = False,
        gumbel_tau: float = 1.0,
        learn_gate_tau: bool = False,
        type: str = "adaptive",
        store_links: bool = False,
        reg_entropy_w: float = 0.0,
        reg_tv_w: float = 0.0,
        pos_weight: float = 1.0,
        vel_weight: float = 0.2,
        # NEW:
        cfot_sparsify: str = "adaptive",
        cfot_keep_mass: float = 0.9,
        cfot_min_k: int = 2,
        cfot_max_k: int | None = None,
        affinity: str = "euclid",
        euclid_scale: float = 1.0,
        cosine_eps: float = 1e-6,
        transport: str = "sinkhorn",
        ):
        super().__init__()
        self.deltas = [int(d) for d in deltas]
        self.D = len(self.deltas)

        # store sparsity knobs
        self.cfot_sparsify = str(cfot_sparsify)
        self.cfot_keep_mass = float(cfot_keep_mass)
        self.cfot_min_k = int(cfot_min_k)
        self.cfot_max_k = (int(cfot_max_k) if cfot_max_k is not None else None)

        self.branches = nn.ModuleList([
            CFOTLayer(
                in_channels=in_channels,
                emb_channels=emb_channels,
                delta=d,
                sinkhorn_iters=sinkhorn_iters,
                sinkhorn_tau=sinkhorn_tau,
                topk=topk,
                pos_weight=pos_weight,
                vel_weight=vel_weight,
                cfot_sparsify=self.cfot_sparsify,
                keep_mass=self.cfot_keep_mass,
                min_k=self.cfot_min_k,
                max_k=self.cfot_max_k,
                affinity=affinity,
                euclid_scale=euclid_scale,
                cosine_eps=cosine_eps,
                transport=transport,
            )
            for d in self.deltas
        ])


        self.store_links = bool(store_links)
        if self.store_links:
            for b in self.branches:
                if hasattr(b, "enable_store_links"):
                    b.enable_store_links(True)

        # Gate over time using simple motion diagnostics
        feat_in = 3 * self.D  # [speed_d, accel_d, res_energy_d] per Δ
        self.gate_net = nn.Sequential(
            nn.Conv1d(feat_in, gate_hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(gate_hidden, self.D, kernel_size=1, bias=True),
        )

        self.hard_gating = bool(hard_gating)
        if learn_gate_tau:
            self.gate_tau = nn.Parameter(torch.tensor(float(gumbel_tau)))
        else:
            self.register_buffer("gate_tau", torch.tensor(float(gumbel_tau)), persistent=True)

        self.reg_entropy_w = float(reg_entropy_w)
        self.reg_tv_w = float(reg_tv_w)

        self.type = str(type).lower()
        self._last_W = None
        self._last_entropy = None
        self._last_tv = None
        self.last_adj_by_delta = None

    # ---- diagnostics features ----
    def _pad_front_1d(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1)
        assert x.dim() == 3
        if pad <= 0:
            return x
        B, C, _T = x.shape
        zeros = x.new_zeros(B, C, pad)
        return torch.cat([zeros, x], dim=2)

    def _speed(self, x: torch.Tensor, d: int) -> torch.Tensor:
        B, C, T, V = x.shape
        if T <= d:
            return x.new_zeros(B, 1, T)
        diff = (x[:, :, d:, :] - x[:, :, :-d, :]).abs().mean(dim=(1, 3))  # [B,T-d]
        return self._pad_front_1d(diff.unsqueeze(1), d)

    def _accel(self, x: torch.Tensor, d: int) -> torch.Tensor:
        B, C, T, V = x.shape
        if T <= 2 * d:
            return x.new_zeros(B, 1, T)
        a = (x[:, :, 2*d:, :] - 2 * x[:, :, d:-d, :] + x[:, :, :-2*d, :]).abs().mean(dim=(1, 3))
        return self._pad_front_1d(a.unsqueeze(1), 2 * d)

    @staticmethod
    def _res_energy(r: torch.Tensor) -> torch.Tensor:
        e = r.abs().mean(dim=(1, 3))  # [B,T]
        return e.unsqueeze(1)         # [B,1,T]

    @torch.no_grad()
    def set_beta(self, b: float):
        for m in self.branches:
            if hasattr(m, "set_beta"):
                m.set_beta(b)

    @torch.no_grad()
    def enable_store_links(self, flag: bool = True):
        self.store_links = bool(flag)
        for b in self.branches:
            if hasattr(b, "enable_store_links"):
                b.enable_store_links(flag)

    def set_gumbel_tau(self, tau: float):
        if isinstance(self.gate_tau, nn.Parameter):
            with torch.no_grad():
                self.gate_tau.copy_(torch.tensor(float(tau), device=self.gate_tau.device))
        else:
            self.gate_tau = torch.tensor(float(tau), device=self.gate_tau.device)

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape

        # per-Δ residuals + motion features
        R_list, feat_chunks = [], []
        for d, layer in zip(self.deltas, self.branches):
            r = layer(x)                         # [B,C,T,V]
            R_list.append(r)
            s = self._speed(x, d)
            a = self._accel(x, d)
            e = self._res_energy(r)
            feat_chunks.extend([s, a, e])

        Fgate = torch.cat(feat_chunks, dim=1)    # [B,3D,T]
        logits = self.gate_net(Fgate)            # [B,D,T]
        if self.hard_gating:
            g_tau = torch.clamp(self.gate_tau, min=1e-2, max=5.0)
            logits_bt = logits.permute(0, 2, 1).contiguous().view(B * T, self.D)
            W_bt = F.gumbel_softmax(logits_bt, tau=float(g_tau), hard=True, dim=1)
            W = W_bt.view(B, T, self.D).permute(0, 2, 1).contiguous()
        else:
            tau = torch.clamp(self.gate_tau, min=1e-2, max=5.0)
            W = F.softmax(logits / float(tau), dim=1)      # [B,D,T]
        self._last_W = W

        R = torch.stack(R_list, dim=1)           # [B,D,C,T,V]
        Rfused = (R * W.view(B, self.D, 1, T, 1)).sum(dim=1)
        Rfused = torch.nan_to_num(Rfused, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            p = (W.float().clamp_min(1e-8)).permute(0, 2, 1)  # [B,T,D]
            self._last_entropy = (-(p * p.log()).sum(dim=-1).mean()).item()
            self._last_tv = ((W[:, :, 1:] - W[:, :, :-1]).abs().mean().item() if T > 1 else 0.0)

        if self.store_links:
            try:
                Ps = []
                for br in self.branches:
                    L = getattr(br, "last_links", None)
                    if isinstance(L, dict) and isinstance(L.get("P", None), torch.Tensor):
                        Ps.append(L["P"].mean(dim=(0, 1)))  # [V,V]
                self.last_adj_by_delta = torch.stack(Ps, dim=0).detach().cpu() if Ps else None
            except Exception:
                self.last_adj_by_delta = None

        return Rfused

    def regularization_loss(self) -> torch.Tensor:
        if self._last_W is None:
            dev = self.gate_tau.device if hasattr(self, "gate_tau") else None
            return torch.tensor(0.0, device=dev) if dev else torch.tensor(0.0)
        W = self._last_W
        p = (W.clamp_min(1e-8)).permute(0, 2, 1)  # [B,T,D]
        ent = -(p * p.log()).sum(dim=-1).mean()
        tv = (W[:, :, 1:] - W[:, :, :-1]).abs().mean() if W.size(-1) > 1 else W.new_tensor(0.0)
        return self.reg_entropy_w * ent + self.reg_tv_w * tv


# ------------------------------------------------------
# Multi-Δ CFOT: parallel branches, fuse residual updates
# ------------------------------------------------------
class MultiDeltaCFOT(nn.Module):
    """
    Parallel CFOT layers at multiple fixed Δ, fused with a 1x1 conv.
    """
    def __init__(self,
        in_channels: int,
        emb_channels: int,
        deltas=(1, 2, 4),
        sinkhorn_iters: int = 5,
        sinkhorn_tau: float = 0.5,
        topk: int = 4,
        beta: float = 1.0,
        clamp: float = 8.0,
        store_links: bool = False,
        pos_weight: float = 1.0,
        vel_weight: float = 0.2,
        # NEW
        affinity: str = "euclid",
        euclid_scale: float = 1.0,
        cosine_eps: float = 1e-6,
        transport: str = "sinkhorn",

    ):
        super().__init__()
        self.branches = nn.ModuleList([
            CFOTLayer(
                in_channels=in_channels,
                emb_channels=emb_channels,
                delta=int(d),
                sinkhorn_iters=sinkhorn_iters,
                sinkhorn_tau=sinkhorn_tau,
                topk=topk,
                beta=beta,
                clamp=clamp,
                pos_weight=pos_weight,
                vel_weight=vel_weight,
                affinity=affinity,
                euclid_scale=euclid_scale,
                cosine_eps=cosine_eps,
                transport=transport,
            )
            for d in deltas
        ])

        self.store_links = bool(store_links)
        if self.store_links:
            for b in self.branches:
                if hasattr(b, "enable_store_links"):
                    b.enable_store_links(True)

        self.fuse = nn.Conv2d(in_channels * len(self.branches), in_channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(in_channels)
        nn.init.kaiming_uniform_(self.fuse.weight, a=math.sqrt(5))

        self.last_adj_by_delta = None

    @torch.no_grad()
    def set_beta(self, b: float):
        for m in self.branches:
            if hasattr(m, "set_beta"):
                m.set_beta(b)

    @torch.no_grad()
    def enable_store_links(self, flag: bool = True):
        self.store_links = bool(flag)
        for b in self.branches:
            if hasattr(b, "enable_store_links"):
                b.enable_store_links(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ups = [b(x) for b in self.branches]         # each [B,C,T,V]
        res = torch.cat(ups, dim=1)                 # [B, C*D, T, V]
        out = self.bn(self.fuse(res))               # [B, C, T, V]
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if self.store_links:
            Ps = []
            for b in self.branches:
                lk = getattr(b, "last_links", None)
                if isinstance(lk, dict) and isinstance(lk.get("P"), torch.Tensor):
                    Ps.append(lk["P"])              # [B,T',V,V]
            if len(Ps) > 0:
                P = torch.stack(Ps, dim=1)          # [B,D,T',V,V]
                self.last_adj_by_delta = P.mean((0, 2)).detach().cpu()  # [D,V,V]

        return out
    
__all__.extend(["SignedAdaptiveDeltaCFOTv2", "SignedMultiDeltaCFOT"])

class _SignedBase(nn.Module):
    def __init__(self, CFOTCls, *, in_channels, emb_channels=64, deltas=(-2,-1,1,2),
                 sinkhorn_iters=5, sinkhorn_tau=0.5, topk=4, store_links=False, **kwargs):
        super().__init__()
        deltas = [int(d) for d in deltas]
        self.pos_d = sorted([d for d in deltas if d > 0])
        self.neg_d = sorted([-d for d in deltas if d < 0])  # magnitudes
        self.store_links = bool(store_links)

        self.pos = None
        self.neg = None
        if self.pos_d:
            self.pos = CFOTCls(in_channels=in_channels, emb_channels=emb_channels,
                               deltas=self.pos_d, sinkhorn_iters=sinkhorn_iters,
                               sinkhorn_tau=sinkhorn_tau, topk=topk,
                               store_links=store_links, **kwargs)
        if self.neg_d:
            self.neg = CFOTCls(in_channels=in_channels, emb_channels=emb_channels,
                               deltas=self.neg_d, sinkhorn_iters=sinkhorn_iters,
                               sinkhorn_tau=sinkhorn_tau, topk=topk,
                               store_links=store_links, **kwargs)

        # exposed for viz
        self.deltas = ([-d for d in self.neg_d] + self.pos_d)
        self.last_adj_by_delta = None  # [D,V,V] averaged
        self._beta_runtime = 1.0

    @torch.no_grad()
    def set_beta(self, b: float):
        self._beta_runtime = float(b)
        for m in (self.pos, self.neg):
            if m is not None and hasattr(m, "set_beta"): m.set_beta(b)

    @torch.no_grad()
    def enable_store_links(self, flag: bool = True):
        self.store_links = bool(flag)
        for m in (self.pos, self.neg):
            if m is not None and hasattr(m, "enable_store_links"): m.enable_store_links(flag)

    def _collect_links(self):
        if not self.store_links:
            self.last_adj_by_delta = None
            return
        chunks = []
        # negative first (signed order)
        if self.neg is not None and getattr(self.neg, "last_adj_by_delta", None) is not None:
            chunks.append(self.neg.last_adj_by_delta)  # [Dn,V,V]
        if self.pos is not None and getattr(self.pos, "last_adj_by_delta", None) is not None:
            chunks.append(self.pos.last_adj_by_delta)  # [Dp,V,V]
        if chunks:
            self.last_adj_by_delta = torch.cat(chunks, dim=0).detach().cpu()
        else:
            self.last_adj_by_delta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,V]
        res = 0.0
        if self.pos is not None:
            res = res + self.pos(x)
        if self.neg is not None:
            x_rev = x.flip(dims=[2])          # reverse time
            r_rev = self.neg(x_rev)           # causal on reversed time
            res = res + r_rev.flip(dims=[2])  # back to original time
        self._collect_links()
        return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


class SignedAdaptiveDeltaCFOTv2(_SignedBase):
    def __init__(self, **kwargs):
        super().__init__(AdaptiveDeltaCFOTv2, **kwargs)
        


class SignedMultiDeltaCFOT(_SignedBase):
    def __init__(self, **kwargs):
        super().__init__(MultiDeltaCFOT, **kwargs)
