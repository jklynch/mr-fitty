from mrfitty.best_subset_selection import BestSubsetSelectionFitTask


def test_get_best_ci_component_count__1():
    """
    Expect to get best_component_count == 1 because all things being equal we want to favor a low component count.
    """
    test_cp_ci_lo_hi = {
        1: (0.1, 0.2),
        2: (0.1, 0.2),
        3: (0.1, 0.2)
    }

    best_component_count = BestSubsetSelectionFitTask.get_best_ci_component_count(test_cp_ci_lo_hi)
    assert best_component_count == 1


def test_get_best_ci_component_count__2():
    """
    Expect to get best_component_count == 2 because the confidence intervals for 2 and 3 overlap.
    """
    test_cp_ci_lo_hi = {
        1: (0.5, 0.6),
        2: (0.3, 0.4),
        3: (0.2, 0.3)
    }

    best_component_count = BestSubsetSelectionFitTask.get_best_ci_component_count(test_cp_ci_lo_hi)
    assert best_component_count == 2


def test_get_best_ci_component_count__3():
    """
    Expect to get best_component_count == 3.
    """
    test_cp_ci_lo_hi = {
        1: (0.5, 0.6),
        2: (0.3, 0.4),
        3: (0.1, 0.2)
    }

    best_component_count = BestSubsetSelectionFitTask.get_best_ci_component_count(test_cp_ci_lo_hi)
    assert best_component_count == 3
