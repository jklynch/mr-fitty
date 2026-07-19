# Development Journal

A running log of development work on MrFitty. Newest entries at the top.

---

## 2026-07-19 12:41 EDT — `plot_holdout_block_structure` visualization

Added `plot_holdout_block_structure` to `notebooks/moving_block_holdout_bootstrap.ipynb`,
along with two small helpers (`contiguous_run_lengths`, `smooth`), a markdown
introduction, and a demo cell. The new cells sit immediately after
`select_holdout_blocks_v5` and *before* the "Development of `select_holdout_blocks`
v1–v5" write-up, so the write-up now has a figure to refer to.

### Motivation

The existing v1–v5 comparison cell measures *outcomes* — holdout frequency per
position, PE histograms, violins, and 95% CIs. But every argument in the v1–v5
write-up is a claim about mask **geometry**: whether the block grid is aligned or
shifted, which positions a shifted grid never reaches, where the circular wrap seam
lands, whether the truncated final block always falls at the high-energy end. The
only geometric evidence in the notebook was a single scalar per version (the holdout
frequency std). This figure plots the masks themselves so those claims can be read
off a plot instead of taken on faith.

It is also cheap: it consumes only `select_holdout_blocks*` output — no NNLS, no
`do_moving_block_holdout_bootstrap` — so it runs in seconds rather than the minutes
the PE comparisons take.

### Layout

One column per version (reusing the label/color convention already established in the
v1–v5 comparison cell: steelblue, darkorange, crimson, mediumseagreen, mediumpurple),
four rows:

1. **Mask rasters + frequency marginal.** Two rasters per column — the low-energy and
   high-energy ends — over a full-width plot of `holdout_masks.mean(axis=0)`, which is
   the column marginal of the very array shown above. Gray vertical lines mark where an
   aligned grid would put its boundaries.
2. **Held-out fraction per iteration.** Shows v1–v3 collapsing to a spike at a constant
   66/198 points while v4/v5 spread over 53–79.
3. **Realized contiguous run lengths.** The *effective* block geometry rather than the
   nominal `block_length`: adjacent selected blocks merge into longer runs, and
   truncated or wrapped blocks appear as short ones.
4. **Holdout vs. resample coverage**, both relative to uniform. The resampling side —
   which blocks remain available after the `valid_block_starts` filter — was not
   plotted anywhere else in the notebook.

### Three revisions the first render forced

- **The raster was illegible.** 198 columns compressed into ~250 px gave ~1.3 px per
  position, so v1's aligned grid did not visibly align — the panel failed at exactly
  the thing it existed to show. Widening the figure could not fix this. Since all the
  interesting differences between versions are *edge* effects, the raster now shows
  the two ends of the spectrum side by side instead of a squeezed whole, at 25
  iterations rather than 50.
- **Run lengths were squeezed** into the first tenth of the axis by a handful of very
  long runs (several selected blocks landing adjacent). The shared x-limit is now the
  99.5th percentile, with the true longest run and the count of runs beyond the axis
  reported in the legend.
- **Row 4 autoscaled per column**, which blew each version's Poisson sampling noise up
  to fill its own axis and made versions that are genuinely flat at 1.0 look as
  structured as v2, whose edges really do collapse. All five columns now share one
  coverage scale. Both curves are also smoothed over one block length; unsmoothed, the
  ~33k resample draws are dominated by counting noise.

`tight_layout` cannot handle the nested `GridSpecFromSubplotSpec` used for the row-1
raster pair, so margins are set explicitly with `subplots_adjust`.

### Verification

- All five holdout frequency stds reproduce the write-up's results table exactly at
  `seed=0`, `n_bootstrap=1000`: 0.0158 / 0.0441 / 0.0122 / 0.0184 / 0.0148.
- `contiguous_run_lengths` checked against a brute-force Python loop over 200 random
  masks, plus all-empty and all-full edge cases. `smooth` verified to leave a constant
  signal constant including at the array edges — it normalizes by a convolution of an
  all-ones array rather than relying on zero padding, which matters because the edges
  are precisely where the versions differ.
- Cells 0–40 executed against the real arsenic data with `UserWarning` promoted to an
  error: clean, 30 axes as expected, and `nbformat.validate` passes.

### Known limitation

v5's alternating mirror is the one claim the figure does not make visually obvious —
it registers only as v5's lower frequency std versus v4, since v5's per-iteration
fraction and run-length distributions are identical to v4's by construction. The other
four versions' distinguishing geometry reads directly off row 1.

---

## 2026-07-03 18:21 EDT — Sum-of-squares reduction in `do_moving_block_holdout_bootstrap`

Investigated, then applied, moving the per-iteration holdout prediction-error (RMSE)
reduction out of the hot loop in `do_moving_block_holdout_bootstrap`
(`notebooks/moving_block_holdout_bootstrap.ipynb`).

### Motivation

After the earlier block-gather vectorization, the per-iteration line
`bootstrap_pes[i] = np.sqrt(np.mean(np.square(holdout_residuals)))` was the largest
remaining pure-Python-overhead cost in the loop. The idea: accumulate each
iteration's holdout residuals and compute all the RMSEs at once after the loop.

### Benchmark setup

Focused benchmark isolating the part the change affects — the bootstrap loop over
all **2,324 reference combinations × 1,000 iterations** (24-reference pool,
`select_holdout_blocks_v3`, seed 42). Full-data fits and holdout draws are
precomputed once and shared, so only `do_moving_block_holdout_bootstrap`'s loop is
timed. 3 repeats each.

### Two candidate forms, and why the rectangular one was rejected

- **Rectangular** (the literal "accumulate residuals in an array"): fill an
  `(n_bootstrap, n_holdout)` array and reduce with
  `np.sqrt(np.mean(np.square(hr), axis=1))`. **Rejected** — it assumes every
  iteration holds out the same number of points. Checked all five selectors: v1/v2/v3
  hold out a constant 66 points, but **v4 and v5 vary (49–80)**. Since the function is
  generic and called with all five, a fixed-width array would break v4/v5.
- **Sum-of-squares** (adopted): accumulate two scalar arrays per iteration —
  `holdout_sum_of_squares[i] = holdout_residuals @ holdout_residuals` and
  `holdout_point_counts[i] = holdout_residuals.shape[0]` — then
  `bootstrap_pes = np.sqrt(holdout_sum_of_squares / holdout_point_counts)` once after
  the loop. Robust to ragged holdout counts, and slightly faster than the rectangular
  form because a single BLAS dot replaces the `np.square` + `np.mean` pair.

### Timing (3 repeats, focused loop over all combinations)

| variant | mean time | speedup |
|---|---|---|
| current (per-iteration `sqrt(mean(square(·)))`) | 26.90s | 1.00× |
| accumulate residuals in rectangular array (constant-count only) | 24.15s | 1.11× (−10%) |
| **accumulate sum-of-squares + counts (adopted)** | **23.09s** | **1.17× (−14%)** |

(±~1s run-to-run variance, so treat as ~10–15%.) The speedup is smaller end-to-end
in `do_ref_subsets_...` because that also spends a few shared seconds on the 2,324
full-data NNLS fits.

### Correctness

`bootstrap_pes` matches the original formula to machine epsilon and coefficients are
bit-identical: max |Δpe| = 1.1e-16, max |Δcoef| = 0. Also verified on **v4** (variable
holdout counts 55–78 in that run): max |Δpe| = 6.9e-18, no NaN/inf — confirming the
ragged-safe path on the case that would have broken the rectangular version.

### line_profiler comparison (233 combinations × 1,000 iterations)

The PE-reduction work dropped from **1.63 s → 0.22 s** (~7.4×):

| | line | per-hit | % of fn |
|---|---|---|---|
| OLD | `bootstrap_pes[i] = np.sqrt(np.mean(np.square(holdout_residuals)))` | 7012 ns | 30.6% |
| NEW | `holdout_sum_of_squares[i] = holdout_residuals @ holdout_residuals` | 712 ns | 4.3% |
| NEW | `holdout_point_counts[i] = holdout_residuals.shape[0]` | 237 ns | 1.4% |
| NEW | `bootstrap_pes = np.sqrt(...)` *(after loop, 233 hits)* | 2471 ns | ~0% |

The single per-iteration line at 7012 ns/hit is replaced by one dot product at
712 ns/hit (~10× cheaper — `r @ r` is a single BLAS call returning a scalar with no
temporary array, versus three dispatched NumPy calls that each allocate/reduce), plus
a trivial `.shape[0]`, with the `sqrt`/divide now run once per call (233×) instead of
once per iteration (233,000×). Everything else is unchanged: NNLS is ~10.5 µs/hit in
both (its *share* rises 46% → 62% only because total time shrank), and the holdout
matmul `A[holdout_mask] @ bootstrap_coef - b[holdout_mask]` is ~3.2 µs/hit in both.
(Profiled *totals* — OLD 5.35 s, NEW 3.82 s — are inflated by line_profiler's per-line
overhead and are only meaningful for *where* time goes; the honest speedup is the
~10–15% from the un-instrumented benchmark above.)

### Result

Adopted the sum-of-squares form in the notebook, using explicit variable names
(`holdout_sum_of_squares`, `holdout_point_counts`) and a comment explaining both the
speedup and the ragged-count-safety rationale. NNLS (~62% of the loop) remains the
dominant cost; further speedup would require reducing or batching the NNLS solves.

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
