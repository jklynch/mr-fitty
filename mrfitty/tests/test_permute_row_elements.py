from collections import Counter

import pandas as pd

from mrfitty.combination_fit import AllCombinationFitTask


def _make_df(data, columns=None):
    return pd.DataFrame(data, columns=columns)


# --- structural properties ---


def test_returns_dataframe():
    """Result is a DataFrame, not an ndarray or None."""
    df = _make_df([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = AllCombinationFitTask.permute_row_elements(df.copy())
    assert isinstance(result, pd.DataFrame)


def test_shape_preserved():
    """Output shape matches input shape."""
    df = _make_df([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = AllCombinationFitTask.permute_row_elements(df.copy())
    assert result.shape == df.shape


def test_columns_preserved():
    """Column labels are unchanged."""
    df = _make_df([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
    result = AllCombinationFitTask.permute_row_elements(df.copy())
    assert list(result.columns) == ["a", "b", "c"]


# --- value preservation per row ---


def test_row_values_are_permutation_of_input():
    """Each output row is a rearrangement of the corresponding input row."""
    df = _make_df(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    result = AllCombinationFitTask.permute_row_elements(df.copy())
    for i in range(df.shape[0]):
        assert sorted(result.iloc[i, :]) == sorted(df.iloc[i, :])


def test_all_identical_row_unchanged():
    """A row where all values are equal is unaffected by shuffling."""
    df = _make_df([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
    result = AllCombinationFitTask.permute_row_elements(df.copy())
    assert list(result.iloc[0, :]) == [5.0, 5.0, 5.0]


def test_single_column_unchanged():
    """A single-column DataFrame has nothing to permute; values must be identical."""
    df = _make_df([[1.0], [2.0], [3.0]])
    result = AllCombinationFitTask.permute_row_elements(df.copy())
    assert list(result.iloc[:, 0]) == [1.0, 2.0, 3.0]


# --- independence of row shuffles ---


def test_rows_shuffled_independently():
    """
    Two rows with the same values should not always receive the same permutation.
    Run many times and confirm the pair of permutations differs at least once.
    """
    row = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _make_df([row, row])
    seen_different = False
    for _ in range(100):
        result = AllCombinationFitTask.permute_row_elements(df.copy())
        if list(result.iloc[0, :]) != list(result.iloc[1, :]):
            seen_different = True
            break
    assert (
        seen_different
    ), "rows always received the same permutation — shuffles are not independent"


# --- shuffle is actually random ---


def test_shuffle_produces_different_outputs():
    """
    Repeated calls on the same input produce different outputs.
    With 5 columns there are 120 permutations; getting the same twice in 50
    tries is astronomically unlikely if the shuffle is truly random.
    """
    df = _make_df([[1.0, 2.0, 3.0, 4.0, 5.0]])
    results = set()
    for _ in range(50):
        result = AllCombinationFitTask.permute_row_elements(df.copy())
        results.add(tuple(result.iloc[0, :]))
    assert len(results) > 1, "shuffle always returned the same permutation"
