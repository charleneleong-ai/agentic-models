"""What each attention scheme costs in memory as the sequence grows.

This is the comparison that motivates the KDA/MLA hybrid, so it lives at the top level rather
than inside whichever layer happened to need it first. The axis that matters is the exponent
on sequence length: MHA and MLA are linear in it, a recurrent state is constant.
"""

from __future__ import annotations

BF16 = 2


def mha_kv_cache_bytes(seq_len: int, n_heads: int, d_head: int, bytes_per_elem: int = BF16) -> int:
    """Vanilla MHA: separate K and V per head, per token."""
    return 2 * seq_len * n_heads * d_head * bytes_per_elem


def latent_kv_cache_bytes(seq_len: int, d_latent: int, bytes_per_elem: int = BF16) -> int:
    """MLA: one latent vector per token, shared across heads, up-projected at attention time."""
    return seq_len * d_latent * bytes_per_elem


def kda_state_bytes(n_heads: int, d_head: int, d_v: int, bytes_per_elem: int = BF16) -> int:
    """KDA: a fixed-size recurrent state. Note the absent `seq_len` — that is the whole point."""
    return n_heads * d_head * d_v * bytes_per_elem
