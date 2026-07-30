r"""Kimi Delta Attention — the delta rule with a channel-wise forget gate.

Reference: Kimi K3 tech report §2.1.1 (Eq. 1-6), building on Kimi Linear and Gated DeltaNet.

The recurrence carries a *fixed-size* state S in R^{d_k x d_v} instead of a growing KV cache:

    S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T        (Eq. 1)
    o_t = S_t^T q_t

Two readings of the same object live here. `kda_recurrent` transcribes Eq. 1 literally,
one token at a time — obviously correct, hopelessly serial. `kda_chunkwise` is the form
that actually runs on a GPU: parallel within a chunk, recurrent across chunks. They agree
to fp32 tolerance (tests/test_kda.py), and that agreement is the whole point — the chunkwise
algebra is where all the subtlety lives.

Derivation of the chunkwise form
--------------------------------
Factor the decay out of the state. With G_i := prod_{r<=i} a_r (cumulative channel decay
from the chunk start) and S_i = Diag(G_i) T_i, Eq. 1 becomes a pure rank-1 update:

    T_i = (I - b_i ktil_i khat_i^T) T_{i-1} + b_i ktil_i v_i^T,
    ktil_i := k_i / G_i,   khat_i := k_i * G_i

so T_i = T_0 + sum_{j<=i} ktil_j u_j^T for row vectors u_j satisfying a triangular system

    (I + A) U = b*V - b*(Khat S),    A_ij = b_i <khat_i, ktil_j> for j < i (strictly lower).

Solving it by forward substitution is the *UT transform*; the result is the paper's
pseudo-value term Vtil = U - W S. Outputs then read the state *after* the current-token
update, so the intra-chunk mask retains its diagonal:

    O = (Q * G) S  +  Tril[(Q * G)(K / G)^T] Vtil                      (Eq. 4)
        \_ inter-chunk _/   \______ intra-chunk ______/

Why the decay is lower-bounded
------------------------------
`K / G` is the reason. G is a product of retention factors in (0,1), so 1/G grows without
bound and overflows in finite precision. Kimi Linear used an unbounded -Softplus log-decay;
K3 bounds it from below with a scaled sigmoid (Eq. 5, `lower_bounded_decay`), so cumulative
log-decay over a 16-token tile stays in (-80, 0) and every tile fits BF16 dynamic range.
That is what lets *all* causal tiles use dense Tensor Core matmuls instead of falling back
to explicit position-pair math on the diagonal.

This implementation never materializes `K / G` at all: it forms the *ratio* G_i / G_j
directly in log space, which is bounded by 1 for every entry the causal mask keeps. That is
numerically safe at any chunk size, but it hides the constraint real kernels live under —
so `lower_bounded_decay` is still the parameterization used, and the tests exercise it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from archlab.attention.heads import GatedOutput, causal_mask, split_heads

G_MIN = -5.0


def lower_bounded_decay(z: Tensor, log_scale: Tensor, g_min: float = G_MIN) -> Tensor:
    """Eq. 5: bound the per-step log-decay from below, then exponentiate to a retention factor.

    Kimi Linear:  g = -exp(A) * Softplus(z)      in (-inf, 0)  — can overflow 1/G
    Kimi K3:      g = g_min * Sigmoid(exp(A) z)  in (g_min, 0) — bounded, Tensor-Core friendly

    Returns alpha = exp(g) in (exp(g_min), 1), channel-wise.
    """
    g = g_min * torch.sigmoid(torch.exp(log_scale) * z)
    return torch.exp(g)


def kda_recurrent(
    q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor, state: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Literal transcription of Eq. 1, one token at a time. The ground truth.

    q, k, alpha: (B, H, T, Dk) | v: (B, H, T, Dv) | beta: (B, H, T) | state: (B, H, Dk, Dv)
    """
    b, h, t, _ = q.shape
    dv = v.shape[-1]
    s = (
        torch.zeros(b, h, q.shape[-1], dv, dtype=q.dtype, device=q.device)
        if state is None
        else state
    )
    outputs = []
    for i in range(t):
        k_i, v_i, a_i = k[:, :, i], v[:, :, i], alpha[:, :, i]
        b_i = beta[:, :, i].unsqueeze(-1)
        s = a_i.unsqueeze(-1) * s  # Diag(a_t) S_{t-1}
        # (I - b k k^T) S  ==  S - b k (k^T S), kept as matrix-vector products
        s = s - b_i.unsqueeze(-1) * k_i.unsqueeze(-1) * torch.einsum(
            "bhc,bhcd->bhd", k_i, s
        ).unsqueeze(-2)
        s = s + (b_i * k_i).unsqueeze(-1) * v_i.unsqueeze(-2)  # + b k v^T
        outputs.append(torch.einsum("bhcd,bhc->bhd", s, q[:, :, i]))
    return torch.stack(outputs, dim=2), s


def chunk_decay_ratio(log_g: Tensor) -> Tensor:
    """exp(logG_i - logG_j) for every (i, j) in a chunk, clamped to the causal half.

    logG is non-increasing in i, so for the entries the causal mask keeps (i >= j) the true
    ratio is already <= 1; clamping the exponent at 0 is exact there and merely prevents the
    masked-out upper triangle from overflowing.
    """
    diff = log_g.unsqueeze(-2) - log_g.unsqueeze(-3)  # (..., C, C, Dk)
    return torch.exp(diff.clamp(max=0.0))


def kda_chunk(
    q: Tensor, k: Tensor, v: Tensor, log_g: Tensor, beta: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    """One chunk of Eq. 3-4: UT transform, then intra- plus inter-chunk output."""
    c = q.shape[-2]
    ratio = chunk_decay_ratio(log_g)  # (B, H, C, C, Dk)
    g = torch.exp(log_g)  # cumulative decay from chunk start, <= 1

    # A_ij = b_i <khat_i, ktil_j>, strictly lower triangular.
    a = (torch.einsum("bhic,bhjc,bhijc->bhij", k, k, ratio) * beta.unsqueeze(-1)).tril(-1)

    # UT transform: solve (I + A) X = RHS by forward substitution. `unitriangular=True`
    # supplies the unit diagonal, so I is never materialized.
    rhs = torch.cat([beta.unsqueeze(-1) * v, beta.unsqueeze(-1) * (k * g)], dim=-1)
    solved = torch.linalg.solve_triangular(a, rhs, upper=False, unitriangular=True)
    u, w = solved[..., : v.shape[-1]], solved[..., v.shape[-1] :]
    v_til = u - torch.einsum("bhic,bhcd->bhid", w, state)  # Vtil = U - W S

    # Output: inter-chunk read of the incoming state + intra-chunk attention over Vtil.
    # diagonal=0 retains self-interaction: token t reads the state after its own write.
    attn = torch.einsum("bhic,bhjc,bhijc->bhij", q, k, ratio) * causal_mask(c, c, q.device)
    out = torch.einsum("bhic,bhcd->bhid", q * g, state) + attn @ v_til

    # Outgoing state, formed so every decay factor stays <= 1.
    log_last = log_g[..., -1:, :]
    k_dec = k * torch.exp((log_last - log_g).clamp(max=0.0))
    new_state = torch.exp(log_last).transpose(-1, -2) * state + torch.einsum(
        "bhjc,bhjd->bhcd", k_dec, v_til
    )
    return out, new_state


def kda_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    state: Tensor | None = None,
    chunk_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """Parallel within a chunk, recurrent across chunks — the form GPUs actually want.

    `chunk_size` is a memory knob, not just a tiling knob: `chunk_decay_ratio` retains an
    O(B*H*T*C*Dk) tensor for backward, linear in C and a factor Dk above the chunk-size-naive
    forms. Comfortable at C<=32 on toy shapes; the dominant allocation by C=128.
    """
    b, h, t, dk = q.shape
    if t % chunk_size != 0:
        raise ValueError(f"sequence length {t} must be divisible by chunk_size {chunk_size}")
    s = (
        torch.zeros(b, h, dk, v.shape[-1], dtype=q.dtype, device=q.device)
        if state is None
        else state
    )

    log_alpha = torch.log(alpha)
    outputs = []
    for start in range(0, t, chunk_size):
        sl = slice(start, start + chunk_size)
        log_g = torch.cumsum(log_alpha[:, :, sl], dim=-2)  # decay from *this chunk's* start
        out, s = kda_chunk(q[:, :, sl], k[:, :, sl], v[:, :, sl], log_g, beta[:, :, sl], s)
        outputs.append(out)
    return torch.cat(outputs, dim=2), s


class ShortConv(nn.Module):
    """Depthwise causal conv over the token dimension, applied to q/k/v projections (Eq. 2)."""

    def __init__(self, dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = x.transpose(-1, -2)
        z = F.pad(z, (self.kernel_size - 1, 0))
        return self.conv(z).transpose(-1, -2)


class KDA(nn.Module):
    """Full KDA layer: Eq. 2 parameterization, Eq. 5 decay, Eq. 6 full-rank output gate."""

    def __init__(
        self, d_model: int, n_heads: int, d_head: int, alpha_rank: int = 16, conv_size: int = 4
    ) -> None:
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        inner = n_heads * d_head

        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.q_conv, self.k_conv, self.v_conv = (ShortConv(inner, conv_size) for _ in range(3))

        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        # Low-rank decay logits, per Eq. 2: z = W_up W_down x + b
        self.alpha_down = nn.Linear(d_model, alpha_rank, bias=False)
        self.alpha_up = nn.Linear(alpha_rank, inner, bias=True)
        self.log_scale = nn.Parameter(torch.zeros(n_heads, 1, 1))  # A_h, initialized to 0

        self.norm = nn.RMSNorm(d_head)
        self.gated_out = GatedOutput(d_model, inner)  # Eq. 6

    def split_heads(self, x: Tensor) -> Tensor:
        return split_heads(x, self.n_heads, self.d_head)

    def forward(self, x: Tensor, chunk_size: int = 32) -> Tensor:
        # q, k: ShortConv -> Swish -> L2Norm.  v: ShortConv -> Swish.
        q = F.normalize(F.silu(self.q_conv(self.q_proj(x))), dim=-1)
        k = F.normalize(F.silu(self.k_conv(self.k_proj(x))), dim=-1)
        v = F.silu(self.v_conv(self.v_proj(x)))
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        z = self.split_heads(self.alpha_up(self.alpha_down(x)))
        alpha = lower_bounded_decay(z, self.log_scale)

        out, _ = kda_chunkwise(q, k, v, alpha, beta, chunk_size=chunk_size)
        out = self.norm(out)  # head-wise RMSNorm before gating
        return self.gated_out(out, x)
