"""The qb-scale runner: the sweep's conclusions are only worth as much as its mechanics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from archlab.ablations.qb_scale import (
    Cell,
    evaluate_cell,
    expand_arms,
    make_scores,
    run_arm,
    steps_to_balanced,
)

CONFIG = Path(__file__).parents[1] / "configs" / "ablations" / "qb-scale.yaml"


@pytest.fixture(scope="module")
def cell() -> Cell:
    return Cell(n_experts=32, n_active=4, n_tokens=512, router_scale=0.15, popularity_skew=0.25)


class TestSyntheticRouter:
    def test_scale_controls_tie_fraction(self) -> None:
        """`router_scale` is the tie axis — the whole saturated arm depends on it working."""
        base = dict(n_experts=64, n_active=4, n_tokens=2048, popularity_skew=0.25)
        mild = make_scores(Cell(**base, router_scale=0.15))
        hot = make_scores(Cell(**base, router_scale=1.0))

        def tie_frac(s: torch.Tensor) -> float:
            return float(((s == 0.0) | (s == 1.0)).float().mean())

        assert tie_frac(mild) == 0.0
        assert tie_frac(hot) > 0.005

    def test_is_reproducible_by_seed(self, cell: Cell) -> None:
        torch.testing.assert_close(make_scores(cell), make_scores(cell))
        assert not torch.equal(make_scores(cell), make_scores(Cell(**{**cell.__dict__, "seed": 1})))

    def test_skew_creates_the_imbalance_being_studied(self, cell: Cell) -> None:
        flat = Cell(**{**cell.__dict__, "popularity_skew": 0.0})
        none = {"id": "none", "balancer": None}
        assert (
            run_arm(none, make_scores(cell), cell, 2)[-1]
            > run_arm(none, make_scores(flat), flat, 2)[-1]
        )


class TestArms:
    def test_null_balancer_never_moves(self, cell: Cell) -> None:
        traj = run_arm({"id": "none", "balancer": None}, make_scores(cell), cell, 4)
        assert len(set(traj)) == 1  # the floor must be a flat line, not a drifting one

    def test_trajectory_length_matches_budget(self, cell: Cell) -> None:
        """Every arm gets the same number of updates — otherwise the comparison is rigged."""
        arms = [
            {"id": "qb", "balancer": "quantile_balance_update"},
            {"id": "sign", "balancer": "sign", "gamma": 1e-2},
            {"id": "alt", "balancer": "alternating_qb_solve", "iters": 5},
            {"id": "none", "balancer": None},
        ]
        for arm in arms:
            assert len(run_arm(arm, make_scores(cell), cell, 6)) == 7  # start + 6 updates

    def test_unknown_balancer_is_rejected(self, cell: Cell) -> None:
        with pytest.raises(ValueError, match="unknown balancer"):
            run_arm({"id": "x", "balancer": "nope"}, make_scores(cell), cell, 1)

    def test_gamma_list_expands_to_one_variant_each(self) -> None:
        """Sign-SGD is swept, not pinned — the fairness guarantee the writeup leans on."""
        assert len(list(expand_arms({"balancer": "sign", "gamma": [1e-3, 1e-2]}))) == 2
        assert len(list(expand_arms({"balancer": "quantile_balance_update"}))) == 1


class TestReporting:
    def test_steps_to_balanced_finds_first_crossing(self) -> None:
        assert steps_to_balanced([2.0, 1.4, 1.02, 1.01]) == 2
        assert steps_to_balanced([2.0, 1.4, 1.2]) is None

    def test_best_variant_is_reported_for_swept_arms(self, cell: Cell) -> None:
        """A swept arm must report its best gamma, not its last."""
        arms = [{"id": "sign-sgd", "balancer": "sign", "gamma": [1e-6, 1e-2]}]
        row = evaluate_cell(cell, arms, steps=4)[0]

        by_gamma = {
            g: run_arm({"balancer": "sign", "gamma": g}, make_scores(cell), cell, 4)[-1]
            for g in (1e-6, 1e-2)
        }
        assert row["load_imbalance"] == pytest.approx(min(by_gamma.values()), abs=1e-4)
        assert row["n_gammas_tried"] == 2


class TestConfigMatchesRunner:
    def test_every_configured_balancer_is_implemented(self) -> None:
        """The spec is committed and the runner reads it — they must not drift apart."""
        cfg = yaml.safe_load(CONFIG.read_text())
        cell = Cell(
            n_experts=32,
            n_active=cfg["sweep"]["n_active"] // 4,
            n_tokens=512,
            router_scale=0.15,
            popularity_skew=cfg["sweep"]["popularity_skew"],
        )
        scores = make_scores(cell)
        for arm in cfg["arms"]:
            for variant in expand_arms(arm):
                assert len(run_arm(variant, scores, cell, 2)) == 3
