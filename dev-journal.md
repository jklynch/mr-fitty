# Development Journal

A running log of development work on MrFitty. Newest entries at the top.

---

## 2026-07-20 20:52 EDT — moving-block length tuned from the data

Added a block-length tuning section to
`notebooks/moving_block_holdout_bootstrap.ipynb` and made the tuned length the default.
Eleven new cells: the estimator and `choose_block_length` inserted before
`do_ref_subsets_moving_block_holdout_bootstrap`, four tests, and a sweep study (function,
six-panel figure, driver, findings) at the end.

### Motivation

Every `select_holdout_blocks` version set `block_length = round(n ** (1/3)) = 6`. That
is the *rate* at which the MSE-optimal moving-block length grows with n, not the length
itself — the constant in front depends on the residual dependence, and the code silently
took it to be 1. The v1–v5 write-up had already flagged that blocks of 6 lose most of the
residual autocorrelation by lag 5 (open question left for separate work). This settles it.

### What was added

- **`politis_white_block_length`** — Politis & White (2004), the standard data-driven
  block length. Returns a dict (b_opt, m_hat, bandwidth, long-run variance, curvature,
  cap) so the number can be audited. Sanity-checked: white noise → 1.16, AR(1) φ=0.8 → 18.3.
- **`choose_block_length`** — reduces the per-combination estimates to one value with a
  **low quantile, not the median**. The holdout draw is shared across all 2,324
  combinations, so one length must serve all. Underfit subsets' residuals contain
  unmodeled spectrum, which the estimator reads as long-range dependence (RMSE–b_opt rank
  correlation +0.54; every M=1 subset pins the cap). That contamination is one-sided —
  it can only inflate the estimate — so the low order statistics are the trustworthy end.
  The p1/p5/p10/p25 sweep prints on every call. Default percentile 10.
- **Four tests**, with their own runner: short blocks for white noise, monotone in φ,
  capped for near-unit-root, and the aggregate surviving structure-dominated rows (the
  one-sided-contamination argument pinned as a test, not a comment).
- **`compare_block_lengths` + `plot_block_length_study`** — sweep L ∈ {3,6,9,10,12,15,20}
  through the whole pipeline with shared seed and design matrix.

### Results

The rule of thumb is too short: p1/p5/p10/p25 = 8.0/10.0/10.3/11.7 over 2,324
combinations, not one asking for 6. **Tuned default is 10.**

But changing it changes no answer. The selected 3-reference subset is `(0,11,19)` at every
L from 3 to 20; the 2-reference subset is `(1,2)` throughout; Spearman ρ of the 2,324
median PEs vs. L=6 never drops below 0.969. The null result is the point — no conclusion
elsewhere in the notebook is an artifact of block length. Both direct diagnostics (ACF
fidelity, bootstrap long-run variance) are *monotone* in L with no interior optimum, so
they only rule 6 out; Politis–White supplies the actual number.

### Scope and knock-on

`do_ref_subsets_moving_block_holdout_bootstrap` now defaults to `block_length='auto'`;
v1–v5 selectors gained an explicit `block_length`. The v1–v5 comparison cells (43/45/49)
are pinned at `block_length=6` — that work is about geometry at the rule-of-thumb length,
and its stored figures and stds (0.0158/0.0441/0.0122/0.0184/0.0148, verified unchanged)
must keep meaning what they said. The interpolation comparison re-ran at L=10: Spearman
moved 0.99992 → 0.99995, same subset selected, so that conclusion is block-length
independent; its findings cell was updated in place with a note. Production
(`mrfitty/prediction_error_fit.py`) is untouched — a separate decision if the tuning is
ever wanted there.

Notebook executes end-to-end clean; 7 existing + 4 new in-notebook tests pass; pytest
unchanged (same 8 pre-existing failures, 50 passed). No `mrfitty/` source modified.

---

## 2026-07-19 21:27 EDT — v1–v5 write-up updated with structural evidence

Added a `### Structural evidence from plot_holdout_block_structure` section to the
"Development of `select_holdout_blocks` v1–v5" markdown cell in
`notebooks/moving_block_holdout_bootstrap.ipynb`, between the results table and the
Analysis section. Pure insertion — no existing text was reworded or removed.

### Motivation

The write-up's comparison rested on two outcome metrics: holdout frequency std and
prediction error with CIs. Its *arguments*, though, are about block geometry, and the
block-structure figure added earlier in the notebook now measures that geometry
directly. Several claims that had been reasoned about could be checked, and a few
things the two summary metrics cannot express could be stated.

All numbers are at `seed=0`, `n_bootstrap=1000`, `n=198` — the same draws the existing
results table is built from — and were re-derived from the selectors rather than read
off the figure.

### Claims confirmed

- **v2's boundary effect.** The shifted grid rarely reaches position 0 or n−1: both
  ends are held out together in **1.5%** of iterations against ~10.7% for v1. This
  accounts for the whole of v2's std penalty (0.0441 vs 0.0158) and confirms it is an
  edge artifact, not a distributed one.
- **v3's wrap-around.** Occurs in **27.3%** of iterations against a predicted
  P(offset > 0) × P(that block held out) = (5/6)(11/33) = 27.8%. v3 is also the only
  version in which a boundary-spanning run is ever a *single* block (12.6%); v1, v2, v4
  and v5 register 0%, their boundary runs being two separately-selected end blocks that
  happen to abut. The physically artificial construct is confirmed unique to v3.
- **v4/v5's wider CIs.** The write-up attributed these to the holdout fraction
  fluctuating when random block lengths do not sum cleanly to n. Confirmed and sized:
  v1–v3 hold out exactly 66 points every iteration, v4/v5 range over **53–79**
  (0.268–0.399 of the data).

### Facts the summary metrics could not show

- **Nominal block length is not realized block length.** v1–v3 all use L=6, but because
  ~1/3 of blocks are selected independently, adjacent selections merge: median run is 6
  as designed, longest runs reach **48, 42 and 36** points. v4/v5 (lengths from [6, 10])
  have median 9, longest 56.
- **v4/v5 leave a larger resample pool** — **52.7%** of block starts available against
  ~47.7% for v1–v3. Both hold out ~1/3 of the data, so this follows from arrangement
  rather than amount: v1–v3 hold out 11 separate blocks of 6 while v4/v5 hold out ~8
  averaging 8, and fewer separate holdout regions contaminate fewer overlapping windows.
  A point in v4/v5's favor that neither existing metric captures, though a ~5 percentage
  point difference is modest.

### A separate question, deliberately scoped out

The reconstructed residual series retains only ~76% of the original lag-1
autocorrelation, ~41% at lag 3, and essentially none by lag 5 — where the original
residuals still carry +0.183. Since `block_length = round(n^(1/3)) = 6` and significant
autocorrelation extends to roughly lag 5–6, L=6 looks marginal for this data.

This applies to **all five versions equally** — they share the same `block_length`
formula and differ only in holdout selection — so it has no bearing on the choice among
them. It is recorded in its own subsection, explicitly flagged as a question about the
block length rather than about v1–v5, and left for a separate investigation.

### Effect on the conclusions

None of the recommendations change, and the section says so explicitly. Quantifying
v3's wrap at more than a quarter of iterations strengthens rather than weakens the
existing argument that v1 is the fallback if the wrap is judged physically
unacceptable. The only finding pointing the other way — v4/v5's larger resample pool —
is real but small and does not offset the variance from their fluctuating holdout
fraction.

---

## 2026-07-19 20:27 EDT — Resampling rows added to `plot_holdout_block_structure`

Extended `plot_holdout_block_structure` in
`notebooks/moving_block_holdout_bootstrap.ipynb` from four rows to eight. The original
four described only the *holdout* draw; the new rows cover the *resample* draw —
`sampled_starts`, the moving blocks pasted together to build each bootstrap iteration's
residual series — and what that resampling actually produces.

### Motivation

Each `select_holdout_blocks` version makes two draws, and only one of them was being
visualized. The resample draw was represented by a single smoothed coverage curve in
row 4 and nothing else, even though it is the half that determines what
`do_moving_block_holdout_bootstrap` feeds to NNLS on every one of its ~2.32M inner
iterations. The two draws are also coupled — holding out a block withdraws every
resample block overlapping it — and that coupling had no picture at all.

### New rows

- **Row 5 — available vs. drawn resample blocks.** Per iteration, each block start is
  classified as unavailable (overlaps a held-out point), available but not drawn, drawn
  once, or drawn 2+ times, since blocks are drawn with replacement. Windowed at both
  ends of the energy range exactly as row 1 is, and for the same reason: a start is one
  or two pixels wide across the full range, and the ends are where the versions differ.
  Each column gets its own legend strip tinted in that column's version color.
- **Row 6 — source position reuse.** How often a source position is reused within one
  iteration. The zero bar (~0.53) sits above the held-out fraction (0.333); the excess
  is the non-held-out positions the draw simply missed.
- **Row 7 — reconstructed vs. original residual series** (only with `residuals=`).
- **Row 8 — autocorrelation.** The substantive one. Block resampling exists to carry
  the residuals' short-range autocorrelation into the bootstrap, and this panel is
  where you can see whether it does: the reconstructed ACF tracks the original out to
  roughly one block length, then collapses past the `block_length` marker. It also sits
  visibly *below* the original at lags 1–5, so the preservation is partial.

`residuals=` is optional and defaults to `None`; without it the function stays purely
geometric, needs no fit, and draws rows 1–6 only (50 axes vs. 60).

### Row 5 was rebuilt once

The first version of row 5 was a *source-position mosaic*: x a position in the
reconstructed series, y the iteration, color the position each value was copied from.
It was correct and it did show the block layout — but it answered "where did this value
come from" when the more useful question is "which blocks could this iteration draw
from, and which did it take". Replaced on that basis. The mosaic's underlying
`source_positions` array is still computed and still drives rows 7–8.

Availability is derived as a cumulative-sum window count — a start `s` is available when
`[s, s + block_length)` contains no held-out point — which restates the condition every
selector applies when it builds `valid_block_starts`, evaluated for all iterations at
once instead of per start.

### Verification

- The block gather used for rows 7–8 reproduces the production gather **exactly**:
  asserted against the original per-iteration
  `np.concatenate([residuals[s:s + block_length] ...])[:n]` form that
  `do_moving_block_holdout_bootstrap` used before it was vectorized. The figure
  therefore shows the real block layout, not an idealization that could drift from the
  code.
- Row 5 asserts that no selector ever draws a resample block overlapping its own
  holdout, rather than drawing a picture that quietly assumes it. Passes for all five.
- Both the with- and without-residuals paths render warning-free; notebook executes
  end-to-end, `nbformat.validate` passes, all seven tests pass; `pytest mrfitty/tests/`
  unchanged at the 8 pre-existing failures / 50 passed.

### Layout note

The figure is now 25×31 inches with residuals. Row heights are driven by a
`height_ratios` list scaled by a fixed inches-per-unit constant, and the margins are
expressed in inches rather than figure fractions, so rows can be added without
squeezing the header or changing how tall each existing row renders.

---

## 2026-07-19 15:44 EDT — Linear vs. cubic spline interpolation of reference spectra

Made the interpolation method a parameter of
`interpolate_references_at_sample_energies` in
`notebooks/moving_block_holdout_bootstrap.ipynb`, and added a permanent in-notebook
comparison of linear against cubic spline interpolation. Notebook-only change;
`mrfitty/base.py` was not touched.

### Motivation

Every reference spectrum is measured on its own energy grid, so all of them are
resampled onto the sample's grid before any fitting happens. That resampling is the
first step in the pipeline, which puts the interpolation method underneath the design
matrix, the NNLS coefficients, the prediction error, and ultimately the reference
combination reported as the best fit. The method had never been examined: it was
whatever `ReferenceSpectrum.__init__` happened to build, an
`InterpolatedUnivariateSpline` (cubic), which
`interpolate_references_at_sample_energies` read off the spectrum object.

### Changes

- New cell defining `make_linear_interpolant` and `make_cubic_spline_interpolant`,
  both built on `scipy.interpolate.make_interp_spline` — the current spline-construction
  API, replacing the legacy `interp1d` / `splrep` interface — so the two methods differ
  only in the degree argument `k` rather than in which function is called.
- `interpolate_references_at_sample_energies` gained a `make_interpolant` parameter
  (default `make_cubic_spline_interpolant`) and now builds the interpolant from each
  reference's own `data_df` rather than reading the pre-built `ref.interpolant`. That
  is what makes the method selectable. The chosen method is printed in the function's
  report, so every downstream figure's output records which interpolation produced it.
- `FakeSpectrum` dropped its now-unused `.interpolant` attribute and the legacy
  `interp1d` import. Two tests added: one that the parameter genuinely selects the
  method (using data with real curvature — every pre-existing fixture uses linear or
  zero data, which *both* methods reproduce exactly and so cannot tell them apart), and
  one pinning the default to cubic.
- New comparison section at the end of the notebook: `compare_interpolation_methods`,
  `summarize_interpolation_comparison`, a six-panel
  `plot_interpolation_method_comparison`, a driver, and a findings write-up.

### Keeping the comparison controlled

Both arms are matched so that the design matrix is the only thing that differs. The
common energy range depends only on the measured ranges of the sample and references,
never on how they are interpolated, so both arms fit the same `n` points and the same
response vector `b`. Because `n` matches, each arm's freshly seeded generator draws
*identical* holdout masks, and every bootstrap iteration trains and scores on exactly
the same positions. All three invariants are asserted at runtime rather than assumed.

### Results

Run on `OTT3_55_spot0.e` against the 24-reference pool, M = [1, 2, 3] (2,324
combinations), 1,000 iterations, `select_holdout_blocks_v3`, seed 42, ~23.5 s per arm.

The design matrices differ substantially — max `|cubic − linear|` = **0.05353** norm
units against a best-fit RMSE of ~0.0280, so the peak disagreement is nearly twice the
error the model is judged by. Despite that, **nothing downstream changes**: Spearman ρ
of median PE across all 2,324 combinations is **0.99992**, the best subset agrees at
every subset size, the ten best combinations are the same ten, and the selected M=3
coefficients sit well inside each other's bootstrap 95% intervals.

Cubic is marginally worse on the typical combination (median Δ = +0.000263) but
marginally better at the optimum (0.028023 vs 0.028235). Small either way, and this
data cannot separate the candidate explanations — recorded as an observation, not a
conclusion.

Recommendation: keep cubic as the default. The valuable result is the negative one —
the hardcoded spline in `mrfitty/base.py` is not quietly steering model selection on
this dataset.

### A premise the data corrected

Going in, the expectation was that the six references measured every 1.05 eV — coarser
than the sample's 0.5 eV grid — would dominate the disagreement. They are indeed
over-represented among the worst offenders, but the single worst is `scorodite`, on an
ordinary 0.50 eV grid. The disagreement is essentially zero above ~11900 eV and
concentrates entirely in 11845–11880 eV, the absorption edge and white line.
**Curvature drives it, not node spacing**: where a spectrum is nearly straight between
nodes a chord and a spline agree however far apart the nodes are, and where it turns
sharply they diverge. The write-up states this rather than the original framing.

This also bounds how far the conclusion travels. The disagreement lives at the edge, so
a reference set measured coarsely *through the edge* — rather than coarsely overall, as
here — could behave differently. Re-running the section is the way to check, which is
why it is a permanent notebook artifact rather than a one-off answer.

### Verification

- The default is non-breaking: on the real 24-reference pool it reproduces what the
  original `ref.interpolant` code computed to **1.9e-15**, so every earlier result in
  the notebook stays reproducible. Separately confirmed the parameter does change
  something (linear differs by 0.05353), so the equivalence is not a no-op.
- All seven tests (five existing, unmodified, plus two new) pass in the notebook
  runner; the full notebook executes and `nbformat.validate` passes at 57 cells.
- `pytest mrfitty/tests/`: 8 failed / 50 passed / 1 skipped — identical to the same run
  on a clean tree, so the pre-existing failures are unrelated to this change.

### Plot revisions worth remembering

Four panels needed fixing after the first render, all the same class of problem as the
previous visualization: an autoscale or default that technically works but communicates
the wrong thing. The zoom showing *why* the methods differ used a ±12 eV window in
which the two curves are visually identical (narrowed to ±2 eV, where the linear chord
across the white-line peak is obvious); the per-reference emphasis colors were sampled
from a colormap slice and one landed on gray, the same color as the de-emphasized
references; the PE scatter on linear axes collapsed every well-fitting combination —
the only ones model selection can choose between — into one corner (now log-log); and
the coefficient bars were grouped by method, so the two methods' values for the same
reference could not be compared side by side (now grouped by reference, over the union
of both selected subsets so it still reads correctly if the methods ever disagree).

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
