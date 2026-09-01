from src.alpha.sticky import StickyLeaderModel

def test_sticky_leader_path_dependent():
    m = StickyLeaderModel()
    assert m.path_dependent is True
    assert m.scores_path_independent is False
    m.reset_trackers()
    assert m._held is None
