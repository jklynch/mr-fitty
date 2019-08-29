"""
The MIT License (MIT)

Copyright (c) 2015-2019 Joshua Lynch, Sarah Nicholas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from mrfitty.best_subset_selection import BestSubsetSelectionFitTask


def test_get_best_ci_component_count__1():
    """
    Expect to get best_component_count == 1 because all things being equal we want to favor a low component count.
    """
    test_median_cp = {1: 0.15, 2: 0.15, 3: 0.15}

    test_cp_ci_lo_hi = {1: (0.1, 0.2), 2: (0.1, 0.2), 3: (0.1, 0.2)}

    best_component_count, _, _ = BestSubsetSelectionFitTask.get_best_ci_component_count(
        test_median_cp, test_cp_ci_lo_hi
    )
    assert best_component_count == 1


def test_get_best_ci_component_count__2():
    """
    Expect to get best_component_count == 2 because the confidence intervals for 2 and 3 overlap.
    """
    test_median_cp = {1: 0.55, 2: 0.35, 3: 0.25}

    test_cp_ci_lo_hi = {1: (0.5, 0.6), 2: (0.3, 0.4), 3: (0.2, 0.3)}

    best_component_count, _, _ = BestSubsetSelectionFitTask.get_best_ci_component_count(
        test_median_cp, test_cp_ci_lo_hi
    )
    assert best_component_count == 2


def test_get_best_ci_component_count__3():
    """
    Expect to get best_component_count == 3.
    """
    test_median_cp = {1: 0.55, 2: 0.35, 3: 0.15}

    test_cp_ci_lo_hi = {1: (0.5, 0.6), 2: (0.3, 0.4), 3: (0.1, 0.2)}

    best_component_count, _, _ = BestSubsetSelectionFitTask.get_best_ci_component_count(
        test_median_cp, test_cp_ci_lo_hi
    )
    assert best_component_count == 3


def test_calculate_prediction_error_list(arsenic_references, arsenic_unknowns):
    """
    """
    a = BestSubsetSelectionFitTask(
        reference_spectrum_list=arsenic_references[:3],
        unknown_spectrum_list=[arsenic_unknowns[0]],
    )

    best_fit, all_counts_spectrum_fit_table = a.fit(arsenic_unknowns[0])
    assert best_fit is not None
