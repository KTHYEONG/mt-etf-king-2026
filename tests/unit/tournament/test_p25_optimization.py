def test_purged_walk_forward_has_no_label_overlap() -> None:
    from src.tournament.optimization import purged_walk_forward_indices

    folds = purged_walk_forward_indices(n_samples=360, horizon=36, n_folds=3)

    assert len(folds) == 3
    for fold in folds:
        assert set(fold.train_indices).isdisjoint(fold.test_indices)
        assert fold.purge >= 36
        assert fold.embargo >= 36
        assert max(fold.train_indices) + fold.purge + fold.embargo < min(fold.test_indices)
