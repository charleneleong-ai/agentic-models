"""A nano language model assembled from an ablation arm's config.

Exists so an ablation varies exactly one object and inherits everything else. Every axis the
configs sweep — attention pattern, depth mixing, activation — resolves to a different module
plugged into the same skeleton, so a loss delta is attributable to that swap and not to two
separately-written models drifting apart.

Small enough to train on CPU in minutes. That is a constraint on what can be *concluded*, not
just on runtime: nano-scale results indicate direction, never magnitude at 2.8T.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from archlab.activations.situ_glu import SiTUGLU
from archlab.attention.kda import KDA
from archlab.attention.mla import MLA
from archlab.depth.attn_res import BlockAttnRes, FullAttnRes
from archlab.depth.residual import ResidualStack
from archlab.mup import BASE_WIDTH, apply_mup_init
from archlab.precision import ActivationStats, Quantizer, identity, quantizer_for


@dataclass
class ModelSpec:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 12
    n_heads: int = 4
    d_head: int = 32
    d_hidden: int = 512
    d_latent: int = 64
    attention_pattern: list[str] = field(default_factory=lambda: ["kda"])
    depth_mixing: str = "residual"
    n_blocks: int = 4
    ffn: str = "situ_glu"
    chunk_size: int = 32
    precision: str = "bf16"
    beta_gate: float = 4.0
    beta_up: float = 25.0
    mup: bool = False  # Standard Parametrization by default; muP for width ladders
    base_width: int = BASE_WIDTH


class SwiGLU(nn.Module):
    """Unbounded baseline for the activation-bound ablation."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)
        self.quantize: Quantizer = identity
        self.stats: ActivationStats | None = None

    def hidden(self, x: Tensor) -> Tensor:
        return nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)

    def forward(self, x: Tensor) -> Tensor:
        h = self.hidden(x)
        if self.stats is not None:
            self.stats.observe(h)  # measured *before* quantization — the true magnitude
        return self.down_proj(self.quantize(h))


class Attention(nn.Module):
    """Uniform interface over KDA and MLA — MLA's cache is irrelevant during training."""

    def __init__(self, kind: str, spec: ModelSpec) -> None:
        super().__init__()
        self.kind = kind
        if kind == "kda":
            self.inner = KDA(spec.d_model, spec.n_heads, spec.d_head)
            self.chunk_size = spec.chunk_size
        elif kind == "mla":
            self.inner = MLA(spec.d_model, spec.n_heads, spec.d_head, spec.d_latent)
        else:
            raise ValueError(f"unknown attention: {kind!r} (expected 'kda' or 'mla')")

    def forward(self, x: Tensor) -> Tensor:
        if self.kind == "kda":
            return self.inner(x, chunk_size=self.chunk_size)
        return self.inner(x)[0]


class Block(nn.Module):
    """Pre-norm attention + FFN, returning the layer's *contribution* rather than h + f(h).

    The outer combination belongs to the depth mixer — that is precisely what the depth-axis
    ablation varies. With `ResidualStack` this reconstructs a standard pre-norm transformer
    exactly; with AttnRes the same contribution is retrieved rather than accumulated.
    """

    def __init__(self, kind: str, spec: ModelSpec) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(spec.d_model)
        self.attn = Attention(kind, spec)
        self.ffn_norm = nn.RMSNorm(spec.d_model)
        if spec.ffn == "situ_glu":
            self.ffn = SiTUGLU(spec.d_model, spec.d_hidden, spec.beta_gate, spec.beta_up)
        elif spec.ffn == "swiglu":
            self.ffn = SwiGLU(spec.d_model, spec.d_hidden)
        else:
            raise ValueError(f"unknown ffn: {spec.ffn!r} (expected 'situ_glu' or 'swiglu')")
        self.ffn.quantize = quantizer_for(spec.precision)

    def forward(self, h: Tensor) -> Tensor:
        a = self.attn(self.attn_norm(h))
        return a + self.ffn(self.ffn_norm(h + a))


def build_depth_mixer(spec: ModelSpec) -> nn.Module:
    if spec.depth_mixing == "residual":
        return ResidualStack(spec.d_model, spec.n_layers)
    if spec.depth_mixing == "full":
        return FullAttnRes(spec.d_model, spec.n_layers)
    if spec.depth_mixing == "block":
        if spec.n_layers % spec.n_blocks:
            raise ValueError(f"n_layers={spec.n_layers} must divide into n_blocks={spec.n_blocks}")
        return BlockAttnRes(spec.d_model, spec.n_blocks, spec.n_layers // spec.n_blocks)
    raise ValueError(f"unknown depth_mixing: {spec.depth_mixing!r}")


class NanoLM(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, spec.d_model)
        pattern = spec.attention_pattern
        self.layers = nn.ModuleList(
            Block(pattern[i % len(pattern)], spec) for i in range(spec.n_layers)
        )
        self.depth = build_depth_mixer(spec)
        self.out_norm = nn.RMSNorm(spec.d_model)
        self.head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)

        # muP: rescale hidden init by 1/sqrt(m) and scale the logits by 1/m, so activation
        # magnitudes and optimal hyperparameters hold still as width changes. Under SP they do
        # not, which makes a width ladder ambiguous between "the mechanism does not transfer"
        # and "the parametrization broke" (Tensor Programs V).
        # Only the init rescale and the per-class learning rates in `mup_param_groups` do work
        # here; muP's output multiplier is absorbed by the RMSNorm before the head, so there is
        # deliberately no logit scaling. See `archlab.mup.output_multiplier`.
        self.width_mult = spec.d_model / spec.base_width if spec.mup else 1.0
        if spec.mup:
            apply_mup_init(self, self.width_mult)

    def forward(self, tokens: Tensor) -> Tensor:
        h = self.depth(self.embed(tokens), list(self.layers))
        return self.head(self.out_norm(h))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def peak_live_sources(self) -> int:
        """What the depth mixer retains — the memory axis AttnRes trades quality against."""
        return self.depth.live_sources()

    def attach_activation_probes(self) -> ActivationStats:
        """Share one accumulator across every FFN.

        Pooled deliberately: the failure SiTU-GLU guards against is a *rare* coincident outlier
        anywhere in the network, so the quantity of interest is the network-wide maximum, not a
        per-layer average that would dilute it.
        """
        stats = ActivationStats()
        for layer in self.layers:
            layer.ffn.stats = stats
        return stats


def losses(logits: Tensor, tokens: Tensor, answer_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Next-token loss split into (local, long-range recall).

    Reported separately because aggregate loss hides *which* capability an arm bought, and
    that attribution is the entire reason to run the ablation.
    """
    pred, target = logits[:, :-1], tokens[:, 1:]
    flat = nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none"
    ).view(target.shape)

    recall = answer_mask[:, 1:]
    local = flat[~recall].mean()
    return local, flat[recall].mean() if recall.any() else torch.zeros((), device=flat.device)
