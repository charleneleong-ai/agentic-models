"""The saturation detector must be able to detect saturation.

`trained_router` reports 0.0000 saturation and 0.0000 ties for every expert count tested, and
concludes from that the qb-scale tie regime is not reached by training. A detector that is simply
broken produces identical output, so the zero readings are only worth something if the detector
is known to fire when saturation is present. That is what these tests pin.
"""

from __future__ import annotations

import pytest
import torch

from archlab.ablations.trained_router import (
    distinct_fraction,
    saturation_fraction,
    tie_fraction,
)

TRAINED_LOGIT_SCALE = 3.0  # upper end of what routers actually reached: |logit|max 2.22 - 3.07


def scores_at(scale: float, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.sigmoid(torch.randn(512, 256) * scale)


class TestDetectorFires:
    """Positive control. Without this, `saturation == 0` is unfalsifiable."""

    @pytest.mark.parametrize(
        ("scale", "min_saturation", "min_ties"), [(10.0, 0.2, 0.9), (30.0, 0.6, 0.99)]
    )
    def test_saturation_and_ties_are_detected(
        self, scale: float, min_saturation: float, min_ties: float
    ) -> None:
        scores = scores_at(scale)
        assert saturation_fraction(scores) > min_saturation
        assert tie_fraction(scores, k=5) > min_ties

    def test_resolution_collapses_as_scale_grows(self) -> None:
        """Distinct-score share is the mechanism behind the ties: fewer values, more collisions."""
        assert distinct_fraction(scores_at(100.0)) < distinct_fraction(scores_at(10.0)) < 1.0


class TestTrainedRegimeIsClear:
    """The gap the conclusion rests on: trained routers sit well below the tied regime."""

    def test_trained_scale_shows_no_ties(self) -> None:
        """Collisions are not quite absent at this scale — roughly 1 score value in 130,000
        repeats — but none of them land on the top-k boundary, which is the only place a
        collision can change routing. The distinction matters: the claim is that training does
        not reach the regime where ties *decide* dispatch, not that float32 never repeats."""
        scores = scores_at(TRAINED_LOGIT_SCALE)
        assert tie_fraction(scores, k=5) == 0.0
        assert distinct_fraction(scores) > 0.9999

    def test_ties_need_a_logit_scale_training_does_not_reach(self) -> None:
        """Locates the onset between the trained scale and the tied regime, so the claim is a
        measured separation rather than an assertion about two hand-picked points."""
        onset = next(s for s in (3, 4, 5, 6, 8, 10) if tie_fraction(scores_at(float(s)), k=5) > 0)
        assert onset > TRAINED_LOGIT_SCALE
