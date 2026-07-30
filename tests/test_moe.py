"""MoE routing: Quantile Balancing must actually equalise load, fast, at 10^3-expert scale."""

from __future__ import annotations

import pytest
import torch

from archlab.moe.latent_moe import LatentMoE, active_param_fraction
from archlab.moe.quantile_balance import (
    alternating_qb_solve,
    expert_load,
    histogram_quantile_bias,
    load_imbalance,
    quantile_balance_update,
    routing_scores,
    topk_route,
)

N_TOKENS, N_EXPERTS, K = 4096, 128, 8


def make_scores(scale: float = 0.15, popularity: float = 0.15) -> torch.Tensor:
    """Router scores with a popularity gradient across experts — the imbalance QB must undo.

    `scale` is kept low deliberately: a saturated sigmoid produces scores of exactly 0.0/1.0,
    and the quantile derivation (Appendix C) assumes no ties. See `TestTieDegradation`.
    """
    torch.manual_seed(0)
    x = torch.randn(N_TOKENS, 64) * scale
    w = torch.randn(N_EXPERTS, 64) * scale
    w += torch.linspace(popularity, -popularity, N_EXPERTS).unsqueeze(-1)
    return routing_scores(x, w)


@pytest.fixture
def skewed_scores() -> torch.Tensor:
    return make_scores()


@pytest.fixture
def alpha(skewed_scores: torch.Tensor) -> torch.Tensor:
    """Per-token Top-k cutoffs under zero bias — the starting point for every QB update."""
    return topk_route(skewed_scores, torch.zeros(N_EXPERTS), K)[2]


def max_bias_err(
    baseline: torch.Tensor, scores: torch.Tensor, a: torch.Tensor, n_bins: int
) -> float:
    approx = histogram_quantile_bias(scores, a, K, n_bins=n_bins)
    return (baseline - approx).abs().max().item()


def balance_after(scores: torch.Tensor, bias: torch.Tensor) -> float:
    routed, _, _ = topk_route(scores, bias, K)
    return load_imbalance(expert_load(routed, N_EXPERTS))


class TestQuantileBalancing:
    def test_unbiased_routing_is_imbalanced(self, skewed_scores: torch.Tensor) -> None:
        """Establish the baseline the fix has to beat: the busiest expert takes ~2x its share."""
        assert balance_after(skewed_scores, torch.zeros(N_EXPERTS)) > 1.8

    def test_single_update_recovers_most_of_the_imbalance(
        self, skewed_scores: torch.Tensor, alpha: torch.Tensor
    ) -> None:
        """One exact quantile jump closes most of the gap — no learning rate, no step budget.

        It does not reach 1.0 in one pass: the update takes the cutoffs alpha as fixed, but
        changing the bias moves them. That is why it is applied every step during training.
        """
        base = balance_after(skewed_scores, torch.zeros(N_EXPERTS))
        after = balance_after(skewed_scores, quantile_balance_update(skewed_scores, alpha, K))
        assert after < 1.25
        assert (base - after) / (base - 1.0) > 0.8  # >80% of the recoverable imbalance, in one step

    def test_beats_fixed_step_sign_updates(self, skewed_scores: torch.Tensor) -> None:
        """Contrast with DeepSeek-V3-style b <- b + gamma*sign(load error), same step budget."""
        bias = torch.zeros(N_EXPERTS)
        target = N_TOKENS * K / N_EXPERTS
        for _ in range(3):
            routed, _, _ = topk_route(skewed_scores, bias, K)
            error = expert_load(routed, N_EXPERTS).float() - target
            bias = bias - 1e-3 * torch.sign(error)
        sign_balance = balance_after(skewed_scores, bias)

        qb_bias = torch.zeros(N_EXPERTS)
        for _ in range(3):
            _, _, alpha = topk_route(skewed_scores, qb_bias, K)
            qb_bias = quantile_balance_update(skewed_scores, alpha, K)
        assert balance_after(skewed_scores, qb_bias) < sign_balance

    def test_alternating_solver_reaches_near_perfect_balance(
        self, skewed_scores: torch.Tensor
    ) -> None:
        """Appendix C, Alg. 1 — the offline reference the online update approximates.

        Alternating the two quantile updates is exact coordinate minimization on the dual, so
        this converges to essentially uniform load, unlike the single deferred step.
        """
        assert balance_after(skewed_scores, alternating_qb_solve(skewed_scores, K, iters=20)) < 1.05

    def test_bias_is_mean_centred(self, skewed_scores: torch.Tensor, alpha: torch.Tensor) -> None:
        """Only relative bias steers routing; centring removes the free constant (Eq. 14)."""
        bias = quantile_balance_update(skewed_scores, alpha, K)
        assert bias.mean().abs() < 1e-5

    def test_bias_does_not_alter_mixture_weights(self, skewed_scores: torch.Tensor) -> None:
        """Eq. 13 omits b from p_ij, so gradients to the router are untouched by balancing."""
        bias = torch.randn(N_EXPERTS) * 0.1
        routed, weights, _ = topk_route(skewed_scores, bias, K)
        torch.testing.assert_close(weights.sum(-1), torch.ones(N_TOKENS))
        # weights are renormalised raw scores of the selected experts — bias-free by construction
        torch.testing.assert_close(
            weights,
            skewed_scores.gather(-1, routed)
            / skewed_scores.gather(-1, routed).sum(-1, keepdim=True),
        )


class TestHistogramEstimator:
    """Appendix D: exact quantiles are unaffordable at scale; binned counts are not."""

    @pytest.mark.parametrize("n_bins", [200, 1000, 4000])
    def test_error_is_bounded_by_bin_width(
        self, skewed_scores: torch.Tensor, alpha: torch.Tensor, n_bins: int
    ) -> None:
        """Appendix D's guarantee: the true quantile and its estimate lie in the same bin.

        Compared against the *order statistic* the estimator targets — the q-th smallest
        required bias, q = mk/n — not against `torch.quantile`, whose linear interpolation
        between neighbouring order statistics is a separate (and, at fine bins, larger)
        discrepancy. See `test_interpolation_floor_dominates_at_fine_bins`.
        """
        required = alpha.unsqueeze(-1) - skewed_scores  # (m, n)

        target = int(N_TOKENS * K / N_EXPERTS)
        order_stat = required.sort(dim=0).values[target - 1]
        exact = order_stat - order_stat.mean()
        approx = histogram_quantile_bias(skewed_scores, alpha, K, n_bins=n_bins)

        bin_width = (required.max() - required.min()).item() / n_bins
        assert (exact - approx).abs().max().item() < 2 * bin_width

    def test_interpolation_floor_dominates_at_fine_bins(
        self, skewed_scores: torch.Tensor, alpha: torch.Tensor
    ) -> None:
        """Refining bins cannot drive error below the rank-discretization gap.

        The histogram lands on an integer rank; `torch.quantile` interpolates between ranks.
        Past a few thousand bins that fixed offset, not the bin width, sets the error floor.
        """
        interpolated = quantile_balance_update(skewed_scores, alpha, K)

        required = alpha.unsqueeze(-1) - skewed_scores
        fine_bin_width = (required.max() - required.min()).item() / 20_000
        err = max_bias_err(interpolated, skewed_scores, alpha, 20_000)
        assert err > fine_bin_width  # floor is rank discretization, not bin width
        assert err < 1e-3  # but still small in absolute terms

    def test_error_shrinks_with_more_bins(
        self, skewed_scores: torch.Tensor, alpha: torch.Tensor
    ) -> None:
        exact = quantile_balance_update(skewed_scores, alpha, K)

        assert max_bias_err(exact, skewed_scores, alpha, 4000) < max_bias_err(
            exact, skewed_scores, alpha, 200
        )

    def test_estimated_bias_balances_as_well_as_exact(
        self, skewed_scores: torch.Tensor, alpha: torch.Tensor
    ) -> None:
        exact = balance_after(skewed_scores, quantile_balance_update(skewed_scores, alpha, K))
        approx = balance_after(
            skewed_scores, histogram_quantile_bias(skewed_scores, alpha, K, n_bins=1000)
        )
        assert approx < exact * 1.05


class TestTieDegradation:
    """The derivation assumes no ties — a saturated router violates that, and it shows."""

    def test_saturated_scores_degrade_balance(self) -> None:
        mild, saturated = make_scores(scale=0.15), make_scores(scale=1.0, popularity=1.0)
        assert (saturated == 1.0).float().mean() > 0.01  # sigmoid pinned at the rail
        assert (mild == 1.0).float().mean() == 0.0

        def solved_balance(s: torch.Tensor) -> float:
            routed, _, _ = topk_route(s, alternating_qb_solve(s, K, iters=20), K)
            return load_imbalance(expert_load(routed, N_EXPERTS))

        assert solved_balance(mild) < 1.05
        assert solved_balance(saturated) > 1.15  # same solver, ties it cannot split


def small_moe(**kwargs: object) -> LatentMoE:
    torch.manual_seed(0)
    return LatentMoE(d_model=32, d_latent=16, d_expert_hidden=24, n_routed=16, n_active=2, **kwargs)


class TestLatentMoE:
    def test_shape_and_sparsity(self) -> None:
        moe = small_moe()
        x = torch.randn(2, 8, 32)
        out, routing = moe(x)
        assert out.shape == x.shape
        assert routing.indices.shape == (2, 8, 2)  # indices keep the token layout of x
        assert moe.sparsity() == 8.0

    def test_forward_does_not_mutate_bias(self) -> None:
        """Balancing is a training-loop step, not a forward side effect."""
        moe = small_moe()
        before = moe.bias.clone()
        moe(torch.randn(4, 16, 32))
        torch.testing.assert_close(moe.bias, before)

    def test_balance_step_takes_effect_on_the_next_forward(self) -> None:
        """A batch must never be routed with a bias derived from itself (causality, §2.3.3)."""
        moe = small_moe()
        x = torch.randn(4, 16, 32)

        _, routing = moe(x)
        moe.balance_step(routing)
        _, after = moe(x)

        assert not torch.equal(routing.indices, after.indices)

    @pytest.mark.parametrize("estimator", [quantile_balance_update, histogram_quantile_bias])
    def test_both_estimators_are_reachable_from_the_module(self, estimator: object) -> None:
        """Appendix D's histogram form is the one that scales — it must not be test-only."""
        moe = small_moe(bias_estimator=estimator)
        x = torch.randn(8, 32, 32)
        _, routing = moe(x)

        moe.balance_step(routing)
        assert torch.isfinite(moe.bias).all()
        assert moe.bias.abs().sum() > 0
        assert moe(x)[0].shape == x.shape

    def test_active_fraction_matches_k3_scale(self) -> None:
        """K3: 16 of 896 routed plus 2 shared — ~2% of expert params per token."""
        assert active_param_fraction(896, 16, 2) == pytest.approx(0.02004, abs=1e-4)
