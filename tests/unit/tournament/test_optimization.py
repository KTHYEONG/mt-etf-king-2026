"""Co-modification coverage for src/tournament/optimization.py (P5)."""

from __future__ import annotations

import pytest

from src.tournament.optimization import purged_walk_forward_indices


def test_purged_walk_forward_indices_partitions_samples() -> None:
    folds = purged_walk_forward_indices(100, 5, 2)

    assert len(folds) == 2
    for fold in folds:
        assert fold.train_indices
        assert fold.test_indices
        assert fold.purge == 5
        assert fold.embargo == 5
        assert max(fold.train_indices) < min(fold.test_indices)


def test_purged_walk_forward_indices_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        purged_walk_forward_indices(0, 5, 2)
    with pytest.raises(ValueError, match="horizon"):
        purged_walk_forward_indices(100, 0, 2)
    with pytest.raises(ValueError, match="n_folds"):
        purged_walk_forward_indices(100, 5, 0)
