# Development Journal

A running log of development work on MrFitty. Newest entries at the top.

---

## 2026-07-03 16:12 EDT — `plot_interpolated_references` visualization

Added `plot_interpolated_references` to `notebooks/moving_block_holdout_bootstrap.ipynb`
(new cells right after the tests) to visualize the output of
`interpolate_references_at_sample_energies`, plus a demo cell that runs it on both
the 3-reference and full 24-reference pools. Key design points:

- **One axis.** The sample response vector `b` (bold black) over the common range,
  the interpolated reference columns of `A`, and the full sample spectrum across
  *all* its energies (dimmed dashed) so the sample points outside the common range
  are visible.
- **Range markers.** Red dashed lines at the exact common-range bounds (recovered
  from the limiting spectra), and gray dotted lines at the first/last sample
  energies actually present in `A`/`b` (which can sit just inside the exact bounds).
- **Scales to many references.** Rather than one colour + legend row per reference
  (which gave a page-wide, hue-repeating legend), references are coloured by the
  *role* they play: low-edge limiters blue, high-edge limiters orange, both-edge
  limiters green, and every other reference a single faint-gray "other references
  (N)" entry. The legend collapses to one row per role with counts; the bound lines
  name the limiting spectra (full names when ≤ 2 tie, otherwise "N spectra"). Colour
  is never the sole cue — roles/counts are spelled out and limiters are named.
- **Readability tuning.** Legend placed outside the axes on the right; dimmed
  context/out-of-range lines darkened; colored reference lines made slightly
  transparent so overlapping references show through.

Verified rendering against both the 3- and 24-reference pools.

---

## 2026-07-03 13:00 EDT — `interpolate_references_at_sample_energies` reporting, return value, and tests

Work on `notebooks/moving_block_holdout_bootstrap.ipynb`, focused on the
`interpolate_references_at_sample_energies` function.

- **Report limiting spectra.** The function computes the usable energy range as
  the intersection of the sample spectrum's range and every reference's range.
  It now identifies and prints which spectrum limits each end — the spectrum with
  the highest lower bound sets the low edge, the one with the lowest upper bound
  sets the high edge. The sample spectrum is included as a candidate limiter.

- **List all tied limiters.** When several spectra share the exact limiting
  energy, every one is reported (comma-separated in the printout), not just the
  first encountered.

- **Return the limiting spectra.** The function's return was extended from
  `(valid_energies, A, b)` to `(valid_energies, A, b, low_limiters, high_limiters)`,
  where the two new values are lists of the limiting spectrum objects. All four
  call sites in the notebook were updated to absorb the extra values via `*_`.

- **Added pytest tests.** New cells immediately after the function definition
  add a minimal `FakeSpectrum` fixture (a lightweight stand-in exposing only
  `.file_name`, `.data_df`, and `.interpolant`) plus five plain pytest functions
  covering: the intersection range and the shapes/values of `A` and `b`;
  identification of the low/high limiters; reporting of tied limiters; the sample
  spectrum acting as the limiter; and identical ranges excluding nothing. A final
  runner cell executes them inside the notebook. The intersection test uses
  reference grids offset a fraction of an eV from the sample grid so the function
  must genuinely interpolate; linear `norm` data keeps the expected values exact.
  All five tests pass.

---

## 2026-07-01 19:11 EDT — Profiling `do_ref_subsets_moving_block_holdout_bootstrap`

### Setup

Profiled the call to `do_ref_subsets_moving_block_holdout_bootstrap` in the last cell of
`moving_block_holdout_bootstrap.ipynb`, using `line_profiler` (`LineProfiler.add_function` on
the target function plus its hot callees, then wrapping the call). The profiled workload is the
notebook's own last-cell parameters: `M=[1, 2, 3]` over 24 filtered reference spectra, giving
2,324 reference combinations x 1,000 bootstrap iterations each (~2.32M inner iterations).

### Baseline findings

`do_ref_subsets_moving_block_holdout_bootstrap` itself was cheap outside of a single loop
(99.4% of its own time) that calls `do_moving_block_holdout_bootstrap` once per reference
combination. All the real cost was inside that inner function's per-iteration loop:

- **33.9%** — `scipy.optimize.nnls(A[train_mask], bootstrap_b[train_mask])`, the NNLS solve
- **22.0%** — `np.sqrt(np.mean(np.square(holdout_residuals)))`
- **26.8%** combined — building `bootstrap_residuals` via a per-iteration Python list
  comprehension over `sampled_starts[i]` followed by `np.concatenate`
- **10.9%** — `A[holdout_mask] @ bootstrap_coef - b[holdout_mask]`

Over 40% of the runtime was Python-level overhead re-gathering blocks on every single one of the
2.32M iterations, even though `sampled_starts` is fully known before the loop starts.

Baseline wall time for the profiled call (including profiler overhead): **~71.8s**.

### Optimization: vectorize block gathering

Replaced the per-iteration list comprehension + `concatenate` with a single fancy-indexing
gather computed once for all bootstrap iterations, before the loop:

```python
block_offsets = np.arange(block_length)
block_indices = sampled_starts[:, :, None] + block_offsets[None, None, :]
all_bootstrap_residuals = residuals[block_indices].reshape(n_bootstrap, -1)[:, :n]
```

Each loop iteration then just indexes the precomputed array: `fitted + all_bootstrap_residuals[i]`.

Correctness was verified against the original per-iteration logic on synthetic data
(`np.allclose` over the gathered residual arrays) before re-profiling.

### Post-optimization findings

Re-profiling `do_moving_block_holdout_bootstrap` after the change:

- Block gathering: **1.4%** of the function's time (down from ~27%), now dominated by the
  vectorized gather done once outside the loop instead of per iteration
- **46.0%** — `scipy.optimize.nnls` (now clearly dominant, as expected for the core solve)
- **29.6%** — `np.sqrt(np.mean(np.square(holdout_residuals)))`
- **14.6%** — `A[holdout_mask] @ bootstrap_coef - b[holdout_mask]`

### Result

Un-profiled wall time for the same last-cell call dropped from **~72s to ~27.1s**, a **~2.6x**
speedup. The function is now dominated by the intrinsic per-iteration work (NNLS solve and the
residual/prediction-error reductions) rather than Python-loop overhead for block resampling.
Further speedup would require reducing the number of NNLS calls or batching the solve itself —
a larger change than this pass covered.

### Investigation: NNLS vs. OLS in `do_moving_block_holdout_bootstrap`

Since `scipy.optimize.nnls` was the single largest cost after the vectorization above (46.0% of
the function's time), investigated whether replacing it with an unconstrained OLS solve (the
same closed-form normal-equations approach as the notebook's `fit_ols` helper) would be faster.

#### Variant tested

Same vectorized-gather structure, with the NNLS call:

```python
bootstrap_coef, _ = scipy.optimize.nnls(A[train_mask], bootstrap_b[train_mask])
```

replaced by:

```python
A_train = A[train_mask]
b_train = bootstrap_b[train_mask]
bootstrap_coef = np.linalg.solve(A_train.T @ A_train, A_train.T @ b_train)
```

#### Timing result: OLS is slower, not faster

Benchmarked with un-instrumented wall-clock time (3 repeats each) of the full
`do_ref_subsets_moving_block_holdout_bootstrap` call (M=[1, 2, 3], 24 refs, 2,324 combinations x
1,000 bootstrap iterations):

| | NNLS (current) | OLS (normal equations) |
|---|---|---|
| mean | **27.71s** | **30.07s** |
| range | 27.49–27.94s | 29.96–30.15s |

OLS was **~8.5% slower** (speedup ratio 0.92x), the opposite of the expected result.

**Why:** `scipy.optimize.nnls` is a single compiled Fortran active-set call. For these small
matrices (max 3 columns, ~130–190 rows) it typically converges in one or two passes, since the
unconstrained solution is often already non-negative for physically well-behaved reference
spectra — so its real cost is close to that of one normal-equations solve. The OLS replacement,
however, does the equivalent work as three separate NumPy calls per iteration
(`A_train.T @ A_train`, `A_train.T @ b_train`, `np.linalg.solve`), each paying its own Python
dispatch and small-array allocation overhead. At this problem's tiny matrix sizes, that per-call
overhead outweighs whatever iteration cost NNLS's active-set loop would otherwise add.

A line profile of the OLS variant confirmed this: the solve-related lines (`A_train = A[train_mask]`
7.8%, `b_train = bootstrap_b[train_mask]` 2.4%, `np.linalg.solve(...)` 49.0%) totaled **59.2%** of
the function's time, versus 46.0% for NNLS's single inline-sliced call — proportionally more
expensive despite doing "less" algorithmically.

(Note: the profiled *absolute* times for the two variants aren't directly comparable — the OLS
loop body has two extra Python statements that `line_profiler` instruments, and that per-line
overhead compounds over 2.32M iterations. The reliable number for absolute timing is the
un-instrumented benchmark above; the profile is only meaningful for where within OLS the time
goes.)

#### It also isn't equivalent

- **25.7%** of the OLS bootstrap coefficients came out negative — physically invalid for this
  XANES linear-combination fit, which is exactly why the notebook uses NNLS.
- Median holdout prediction error differed by an average of **4.6%** (relative) per combination
  between the two methods (mean of median PE: 0.206 for OLS vs. 0.195 for NNLS) — OLS's
  unconstrained fits generalize slightly worse on held-out data here.

#### Conclusion

Replacing NNLS with OLS in `do_moving_block_holdout_bootstrap` would be a net loss on both axes:
slightly slower *and* it produces physically invalid (negative) coefficients and measurably
different prediction-error estimates. Not recommended — NNLS should stay.
