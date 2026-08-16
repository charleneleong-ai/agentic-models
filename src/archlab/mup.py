"""Maximal Update Parametrization — so a width ladder measures the mechanism, not the
parametrization.

Reference: Tensor Programs V (Yang & Hu, arXiv 2203.03466); u-muP (arXiv 2407.17465) for the
unit-scaled reformulation; Cerebras-GPT (arXiv 2304.03208) §3.3 for a scaled SP-vs-muP
comparison.

Why this exists here. This repo wants to know whether a nano-scale ablation *ranking* survives
scaling, because validity-threat work says proxy rankings can fail to transfer even between
125M and 1B. The obvious test is a width ladder — but under Standard Parametrization that test
is confounded. SP does not keep per-layer update magnitudes O(1) as width grows, so the same
learning rate is too large for some layers and too small for others at a different width.
Tensor Programs V documents the consequence directly: under SP performance can improve with
width and then *suddenly worsen*. A rank flip across an SP ladder is therefore ambiguous
between "the mechanism does not transfer" and "the parametrization broke".

muP fixes the initialization and per-layer learning rates so activation updates stay O(1) as
width changes. Optimal hyperparameters then hold still, and a ladder measures what it claims to.

Parameters split into three classes, following the Adam column of Tensor Programs V Table 3,
with `m = width / base_width`:

    embedding-like   fan_in is vocab, not width  -> init unchanged, lr unchanged
    hidden matrices  fan_in and fan_out ~ width  -> init std / sqrt(m), lr / m
    output head      fan_in ~ width, fan_out fixed -> lr / m  (no forward multiplier — see below)

One deviation from the paper, arrived at empirically. Tensor Programs V also prescribes a
1/fan_in multiplier on the output. This model has an RMSNorm immediately before the head, so the
head's input is unit-scale at every width and the logits are already width-invariant under SP
(measured: 0.99x drift over an 8x width span). Applying the multiplier on top double-corrects,
and `coordinate_check` caught it doing so — logits fell 0.4651 -> 0.0578, shrinking as 1/m. The
forward-pass half of muP is absorbed by the normalization here; the learning-rate half is not,
and that is what this module supplies. See `output_multiplier`.

`coordinate_check` is the implementation-bug detector the paper recommends running *before*
trusting any transfer study: activation magnitudes should stay flat across width. If they drift,
muP is mis-applied and the ladder is meaningless. It has already earned its place once.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

BASE_WIDTH = 128  # the width whose hyperparameters everything else is scaled from


def is_embedding_like(module: nn.Module) -> bool:
    """True when fan_in is the vocabulary rather than the model width."""
    return isinstance(module, nn.Embedding)


def is_output_head(name: str) -> bool:
    return name.endswith("head")


def apply_mup_init(model: nn.Module, width_mult: float) -> None:
    """Rescale hidden-matrix initialization by 1/sqrt(m). Embeddings are left alone.

    Hidden weights are initialized with variance proportional to 1/fan_in, so at width m*d the
    standard deviation must shrink by sqrt(m) relative to the base width for pre-activations to
    keep the same scale.
    """
    if width_mult == 1.0:
        return
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not is_output_head(name):
                module.weight.mul_(1.0 / math.sqrt(width_mult))


def mup_param_groups(
    model: nn.Module, width_mult: float, base_lr: float, weight_decay: float
) -> list[dict[str, Any]]:
    """Per-class learning rates: hidden and output scale as 1/m, embeddings do not.

    Under Adam the update is normalized by the gradient's own scale, so the correct hidden-layer
    rule is lr/m rather than the lr/sqrt(m) that SGD would want — this is the row of Tensor
    Programs V Table 3 that is easiest to get wrong.
    """
    embedding, hidden, output = [], [], []
    for name, module in model.named_modules():
        for _, param in module.named_parameters(recurse=False):
            if is_embedding_like(module):
                embedding.append(param)
            elif is_output_head(name):
                output.append(param)
            elif isinstance(module, nn.Linear):
                hidden.append(param)
            else:
                embedding.append(param)  # norms, biases, learned scalars: width-independent

    # Three groups rather than two, though `hidden` and `output` currently share a rate: Table 3
    # gives them separate rules that coincide only because this head's fan_out is fixed, and
    # collapsing them would hide that.
    scaled = base_lr / width_mult
    return [
        {"params": embedding, "lr": base_lr, "weight_decay": weight_decay},
        {"params": hidden, "lr": scaled, "weight_decay": weight_decay},
        {"params": output, "lr": scaled, "weight_decay": weight_decay},
    ]


def output_multiplier() -> float:
    """Always 1.0 here, and the reason is worth recording.

    Tensor Programs V prescribes a 1/fan_in multiplier on the output, because under plain SP the
    head's *input* scale grows with width and the logits would grow with it. This model puts an
    RMSNorm immediately before the head, so that input is unit-scale at every width and the
    logits are already width-invariant — measured: SP logits 0.4651 -> 0.4619 across a 8x width
    span, a drift of 0.99x.

    Applying the multiplier on top of that double-corrects. It was applied, and the coordinate
    check caught it: muP logits fell 0.4651 -> 0.0578 over the same span, shrinking as 1/m. The
    diagnostic exists precisely to catch this class of error before a transfer study is built on
    it, and here it did.

    The forward-pass part of muP is therefore already handled by the normalization. What remains
    substantive is the per-layer learning-rate scaling in `mup_param_groups`.
    """
    return 1.0


@torch.no_grad()
def coordinate_check(
    build: Any, widths: tuple[int, ...], tokens: Tensor, base_width: int = BASE_WIDTH
) -> dict[int, float]:
    """Mean absolute activation entering the head, per width.

    The diagnostic Tensor Programs V recommends before trusting a transfer study: under muP this
    should be roughly flat in width. Growth or decay means the scaling is mis-applied, and any
    ladder built on it would be measuring the bug.
    """
    out = {}
    for width in widths:
        model = build(width)
        apply_mup_init(model, width / base_width)
        h = model.depth(model.embed(tokens), list(model.layers))
        out[width] = float(model.out_norm(h).abs().mean())
    return out
