# Development Journal

A running log of development work on MrFitty. Newest entries at the top.

## Contents

<!-- toc -->
- [2026-09-12 15:50 EDT — subsets that tie the best one, and what "tie" has to mean](#2026-09-12-1550-edt--subsets-that-tie-the-best-one-and-what-tie-has-to-mean)
- [2026-09-11 21:21 EDT — two coverage rows were one plot, and the block structure figure is returned](#2026-09-11-2121-edt--two-coverage-rows-were-one-plot-and-the-block-structure-figure-is-returned)
- [2026-09-11 19:51 EDT — `plot_bootstrap_summary` reports medians, and hands back its figures](#2026-09-11-1951-edt--plot_bootstrap_summary-reports-medians-and-hands-back-its-figures)
- [2026-09-11 15:30 EDT — a fit file now says which mrfitty wrote it, and when](#2026-09-11-1530-edt--a-fit-file-now-says-which-mrfitty-wrote-it-and-when)
- [2026-09-08 17:12 EDT — a fit in one file, and a table you can ask questions of](#2026-09-08-1712-edt--a-fit-in-one-file-and-a-table-you-can-ask-questions-of)
- [2026-09-08 15:01 EDT — the cluster cutoff is coarse because the references are one family](#2026-09-08-1501-edt--the-cluster-cutoff-is-coarse-because-the-references-are-one-family)
- [2026-09-08 11:11 EDT — reference clustering, and the trees folded into the fit summaries](#2026-09-08-1111-edt--reference-clustering-and-the-trees-folded-into-the-fit-summaries)
- [2026-09-07 12:25 EDT — pipeline overview at the top of the bootstrap notebook](#2026-09-07-1225-edt--pipeline-overview-at-the-top-of-the-bootstrap-notebook)
- [2026-09-06 22:32 EDT — `ssr` renamed to `rss_residuals`](#2026-09-06-2232-edt--ssr-renamed-to-rss_residuals)
- [2026-09-06 22:10 EDT — three bugs behind eight failing tests](#2026-09-06-2210-edt--three-bugs-behind-eight-failing-tests)
- [2026-09-06 21:24 EDT — the v1–v5 ranking was a statement about `n mod block_length`](#2026-09-06-2124-edt--the-v1v5-ranking-was-a-statement-about-n-mod-block_length)
- [2026-07-20 21:03 EDT — auto-generated table of contents for this journal](#2026-07-20-2103-edt--auto-generated-table-of-contents-for-this-journal)
- [2026-07-20 20:52 EDT — moving-block length tuned from the data](#2026-07-20-2052-edt--moving-block-length-tuned-from-the-data)
- [2026-07-19 21:27 EDT — v1–v5 write-up updated with structural evidence](#2026-07-19-2127-edt--v1v5-write-up-updated-with-structural-evidence)
- [2026-07-19 20:27 EDT — Resampling rows added to `plot_holdout_block_structure`](#2026-07-19-2027-edt--resampling-rows-added-to-plot_holdout_block_structure)
- [2026-07-19 15:44 EDT — Linear vs. cubic spline interpolation of reference spectra](#2026-07-19-1544-edt--linear-vs-cubic-spline-interpolation-of-reference-spectra)
- [2026-07-19 12:41 EDT — `plot_holdout_block_structure` visualization](#2026-07-19-1241-edt--plot_holdout_block_structure-visualization)
- [2026-07-03 18:21 EDT — Sum-of-squares reduction in `do_moving_block_holdout_bootstrap`](#2026-07-03-1821-edt--sum-of-squares-reduction-in-do_moving_block_holdout_bootstrap)
- [2026-07-03 16:12 EDT — `plot_interpolated_references` visualization](#2026-07-03-1612-edt--plot_interpolated_references-visualization)
- [2026-07-03 13:00 EDT — `interpolate_references_at_sample_energies` reporting, return value, and tests](#2026-07-03-1300-edt--interpolate_references_at_sample_energies-reporting-return-value-and-tests)
- [2026-07-01 19:11 EDT — Profiling `do_ref_subsets_moving_block_holdout_bootstrap`](#2026-07-01-1911-edt--profiling-do_ref_subsets_moving_block_holdout_bootstrap)
<!-- /toc -->

---

## 2026-09-12 15:50 EDT — subsets that tie the best one, and what "tie" has to mean

`plot_best_subset_bootstrap_summaries` ranks reference subsets by median holdout prediction
error and labels the top three 1st, 2nd and 3rd best. That is a point estimate and says
nothing about whether the gaps are real. `notebooks/moving_block_holdout_bootstrap.ipynb`
gains `plot_best_peci_subset_bootstrap_summaries`, which reports the subsets whose prediction
error cannot be distinguished from the lowest instead, and two studies that had to be run
before it could be written.

### The comparison is paired, and that is the whole reason it works

`do_ref_subsets_moving_block_holdout_bootstrap` calls `select_holdout_blocks_fn` once and
scores every combination on those draws, so on iteration k every subset predicted the same
held-out energies. `bootstrap_pes[i] - bootstrap_pes[best]` is therefore a paired difference:
an unlucky holdout that lands on the whiteline hurts both subsets at once and cancels.

Comparing each subset's own interval against the best subset's interval for overlap throws
that away, and on this data it would call nearly everything tied. Two subsets can have
thoroughly overlapping intervals while one loses on every single iteration.
`test_peci_paired_comparison_is_sharper_than_overlapping_intervals` is that construction.

### Which interval estimator: they agree, but one of them often returns nothing

Three things in this repository are called a 95% CI — BCa on the median
(`mrfitty/prediction_error_fit.py`), raw percentiles of the draws
(`mrfitty/bootstrap_validation_fit.py`), and the percentile bootstrap of the median
(`bootstrap_ci`, already in the notebook for the v1–v5 comparison). The new section measures
all three against a median known exactly, resampling an observed 1,000-draw array as its own
population so the real right-skewed shape is preserved and nothing has to be assumed.

They agree: intervals differ in the fourth significant figure, widths to about 1%, and no
tied/not-tied verdict changes because of the estimator. Coverage is 94–95% for the percentile
bootstrap and 95–96% for the order statistic, both inside the ±1% noise of a 500-replicate
estimate.

`scipy.stats.bootstrap` returned NaN on 5.6% of the prediction error arrays and 16.6% of the
paired differences. BCa estimates an acceleration constant by jackknifing, and that estimate
divides by zero when the jackknife values are all equal. A bootstrap prediction error array is
full of ties — each entry is an RMSE over a resampled set of held-out points — so the median
often does not move when one point is dropped. The degenerate case is the best subset against
itself, a column of exact zeros, which fails every time; `peci_tie_table` special-cases that
comparison rather than trusting any estimator with it.

So the choice came down to cost, and the order statistic is about ten thousand times cheaper
for the same answer. It is the default.

### The rule that decides ties was the real question

Putting a confidence interval on the median paired difference — the obvious reading of
"compare the prediction error confidence intervals", and what
`PredictionErrorFitTask.get_best_ci_component_count` does for component counts — does not
work here. Its width falls as `1 / sqrt(n_bootstrap)`, so the tie set shrinks as the bootstrap
runs longer while the data behind it never changes:

| n_bootstrap | 50 | 100 | 250 | 500 | 1000 |
|---|---|---|---|---|---|
| CI on the median difference (M=3) | 3 | 2 | 1 | 1 | 1 |
| middle 95% of the differences (M=3) | 12 | 14 | 6 | 6 | 6 |

At 1,000 iterations it left exactly one subset at every subset size, which reduces the new
function to what the old one already reported. Its answer is partly a statement about a knob
the analyst set.

Bracketing the middle 95% of the differences themselves asks instead how often the two
subsets change places. That spread is a property of the spectrum and the references, it does
not move with the iteration count, and on the same fit it gives tie sets of 2, 9 and 6.
`tie_by_paired_distribution` is the default; `tie_by_paired_median_ci` is one argument away
and kept, because its behavior here is a reason to look again at the component count
selection in `prediction_error_fit.py`, which has the same dependence on an iteration count.

The default is lenient — a subset that loses 90% of the draws still straddles zero — so
`peci_tie_table` also records `d_win_rate`, the fraction of iterations a subset beat the best
one, which is the same comparison with no interval convention in the way.

### Ranking by median prediction error is not ranking head to head

The `d_win_rate` column made an inversion visible immediately. At M=2 the subset ranked third
by median prediction error has a *negative* median paired difference and predicts the held-out
energies better than the nominal winner on 64% of iterations. The two orderings are different
questions, and only the paired comparison shows it.

### Shape of the new code

`peci_tie_table` is pure computation and returns one row per combination, following the
separation `best_subsets_by_size` already keeps. `plot_best_peci_subset_bootstrap_summaries`
takes that table as an argument rather than computing it, and returns its figures closed, the
treatment `plot_bootstrap_summary` and `plot_holdout_block_structure` already got.
`max_subsets_per_size` is the only thing bounding the output: a tie set can hold hundreds of
subsets and each costs four or five figures.

Nine tests, run by a driver at the foot of the cell. Two of them are arguments rather than
numbers — that pairing beats overlap, and that one tie rule sharpens with iterations while the
other does not. The second asserts interval *widths*, not tie counts: a verdict flips only
when the width crosses the offset, and where that happens for any one draw is luck.

Verified against a real 92-combination fit from the arsenic sample data rather than the full
2,324-combination search, which is a long job; the counts quoted in the Findings cells are
stated as trends for that reason. `pytest mrfitty/tests/` is unchanged at 58 passed, 1
skipped.

---

## 2026-09-11 21:21 EDT — two coverage rows were one plot, and the block structure figure is returned

`plot_holdout_block_structure` in `notebooks/moving_block_holdout_bootstrap.ipynb` had nine
rows; it has eight, because two of them were drawing the same quantity.

### The same curve twice

Row 3 was `plot_holdout_frequency_marginal`: the fraction of iterations that held out each
position, against energy. Row 4 was `plot_coverage_relative_to_uniform`, whose holdout curve
was that same array divided by its mean and smoothed over one block length, drawn against the
same energy axis directly beneath it. Two rows, one quantity, differing in a constant scale
factor and a smoothing window.

They are now one row. What each had that the other did not is kept:

- **The absolute scale.** Row 4 could only say "1.3× as often as uniform"; row 3 could say
  "held out in 39% of iterations". A `secondary_yaxis` on the right restates the left axis in
  absolute terms, so the panel answers *how evenly* and *how much* at once. It takes forward
  and inverse transforms rather than fixed limits, so it follows the shared y-limits without
  being told about them. It calibrates the holdout curve only — the resample curve is divided
  by its own mean, which is a count of block starts and not a fraction of anything.
- **The unsmoothed curve.** The merged panel draws the holdout frequency per position, as row
  3 did. The resample curve is still smoothed, because unsmoothed it is Poisson noise from
  tens of thousands of draws with no readable structure, but the holdout curve is not: its
  wobble is mostly the sampling noise of `n_bootstrap` Bernoulli draws per position, and that
  noise is the scale any apparent structure has to be judged against. Smoothing it away
  invites reading a version as flat when it is merely quiet. Only one of the two curves is
  smoothed, so the title now says which.
- **The uniformity statistic.** The mean and std of the unsmoothed frequency are in the
  legend. That std is what the v1–v5 results table is built from.

Dropping the smoothed holdout curve meant the shared y-limits had to be recomputed from the
unsmoothed one. They were taken over the smoothed curve, which is narrower, and left as they
were would have clipped what the panel actually draws.

The removal is not only a tidier figure: v1's high-energy end is a step down to zero, and the
smoothed curve had been drawing it as a gentle ramp.

### The figure is returned, not shown

`plot_holdout_block_structure` ended in `plt.show()`. It now returns the one figure it builds,
closed with `plt.close` first — the inline backend draws every figure still open at the end of
a cell, so a returned-but-open figure appears twice, once from that flush and once from the
caller. The caller is `display(plot_holdout_block_structure(...))`. Same treatment
`plot_bootstrap_summary` got in the previous entry.

### Two layout repairs the new axis forced

Neither was optional; the merged row is unreadable without them.

- `wspace` 0.25 → 0.42. The right-hand axis label ran into the next column's left label.
- The right margin was a flat `right=0.99`, which left the rightmost column's tick labels,
  axis label and title off the edge of the figure once that column had an axis on its right.
  It is now `1 - 0.75 / figure_width`, in inches like the top and bottom margins, so it holds
  as columns are added.

Both were found by rendering the figure and looking at it, not by reading the code. Executing
the whole notebook takes about twelve minutes; the figure alone, from a script that runs the
cells it depends on, takes twenty seconds, which is what made looking twice cheap enough to
do.

---

## 2026-09-11 19:51 EDT — `plot_bootstrap_summary` reports medians, and hands back its figures

Three changes to one function in `notebooks/moving_block_holdout_bootstrap.ipynb`. The first
changes what the figures say; the second changes how they are used.

### The blue marker is now the median

Every bootstrap distribution in these figures was summarized by its mean — a blue dashed line
on each histogram, a blue diamond on each violin. All of them are medians now.

The mean was the wrong statistic for both distributions on the figure, for the same reason in
each case: they are bounded below and lean right, so the mean sits above the bulk of the draws
and reports a typical value the distribution rarely takes. The prediction errors are RMSEs, so
zero is a hard floor and the tail runs upward. The coefficients pile up *against* zero, because
the fit is non-negative least squares and a reference the fit does not want is pushed to exactly
zero rather than to a small negative number — so a reference that the bootstrap zeroes out in
most draws and gives real weight in a few gets a mean that sits well above the zero it usually
takes.

The other reason is that the median is what the combination search already ranks on — the
"best" subset is the one with the lowest median prediction error, `median_pe` is the column
written to the Parquet file, and `plot_descending_median_pe` plots medians. The blue marker was
the one place reporting something else. It is now the same statistic that chose the combination
the figure is about.

One mean stays: `rmse = np.sqrt(np.mean(residuals ** 2))`. That mean is the definition of RMSE,
not a summary of a bootstrap distribution, and the marker it feeds is labeled `RMSE=`. It is
commented as such so the next reader does not tidy it.

### The function returns figures instead of showing them

`plot_bootstrap_summary` called `plt.show()` four times, so drawing was all it could do. It now
returns its figures as a tuple in reading order — the fit, the reference trees, the residual
diagnostics, the coefficient histograms, the coefficient violins — four of them, or five when
`clusterings` is given. Callers do `for fig in plot_bootstrap_summary(...): display(fig)`, which
is what both of them now do.

Each figure is closed with `plt.close` before being returned, and that is load-bearing rather
than tidiness: the inline backend draws every figure still open at the end of a cell, so a
returned-but-open figure appears twice, once from that flush and once from the caller. Closing
discards nothing — a closed figure still renders when displayed, saved, or edited further.

The re-run confirms the arithmetic: the single-fit cell emits 4 figures, and the cell running
the full search emits 48 — 3 from `plot_ref_subsets_summary` plus 9 combinations × 5 — all 48
distinct by content hash, so nothing is drawn twice.

### Docstring

Parameters and Returns sections, numpydoc style, matching `write_fit_results`. Shapes for every
array, `bootstrap_coefs` noted as ordered to match `ref_names` and `coef`, and the `fitted - b`
residual convention spelled out. The existing prose about why each row is its own figure is
kept above them.

---

## 2026-09-11 15:30 EDT — a fit file now says which mrfitty wrote it, and when

The Parquet file from the last entry recorded everything about the fit and nothing about the
run that produced it. Open one six months from now and it cannot tell you whether its numbers
came from the code you have checked out today. Two fields in the file's key-value metadata fix
that:

- **`mrfitty_version`** — `mrfitty.__version__`, the versioneer string, so the entry is a
  commit and not just a release: `0.14.0.post68.dev0+g37574c4`. That is what tells a reader
  whether a file predates a change in how the prediction errors were computed.
- **`written_at`** — `datetime.now(timezone.utc).isoformat()`. UTC with an explicit offset, so
  the timestamp means the same thing to a reader in another timezone and sorts as text.

Both ride in the same JSON metadata document as the rest of the provenance, so reading them
still touches no data — not one row, not one column. `read_fit_results` returns them alongside
the seed and block length, and query 1 in the notebook's "six things the file answers" cell
prints them.

### Old files still open

`RESULTS_SCHEMA_VERSION` goes 1 → 2, and the reader gained a separate
`READABLE_SCHEMA_VERSIONS = (1, 2)`. The writer only ever writes the current version; the
reader accepts every version in that tuple, so a file written before these two fields existed
still opens and the fields come back as `None` — *this file does not say*, rather than a
`KeyError` or a refusal. The returned dict has the same keys at every version, which is the
point: a caller reads `mrfitty_version` without first asking what version the file is.

Refusal is kept for the case that deserves it. A version outside the tuple is one this code
cannot reconstruct, not one missing a label, and it still raises.

### Tests

Three, all in the same cell, nine there now and all passing:

- `test_the_file_says_what_wrote_it_and_when` — the version matches the running package, and
  the timestamp is parsed with `fromisoformat` rather than compared as text, so a malformed
  one fails here rather than in whatever reads the file later. It checks the parsed time
  carries a `tzinfo` and falls between the two clock readings that bracket the write.
- `test_a_version_1_file_still_reads` — a version 1 file is manufactured by `_rewrite_metadata`,
  which copies a results file with its metadata document edited and the table untouched, so
  what the reader sees differs from a current file exactly where a real old file would. Asserts
  everything version 1 carried round-trips unchanged and both new fields come back `None`.
- `test_an_unreadable_schema_version_is_still_refused` — the same helper, stamped one past the
  highest readable version.

All in `notebooks/moving_block_holdout_bootstrap.ipynb`.

---

## 2026-09-08 17:12 EDT — a fit in one file, and a table you can ask questions of

Every figure in the notebook was drawn from a live fit, so redrawing any of them — after
changing a plot, or to look again at last week's run — meant re-running the
2,324-combination bootstrap. Nothing about a finished fit survived the kernel.

`write_fit_results` now puts a fit in one Parquet file and `read_fit_results` hands back
exactly the dict `do_fits_and_plot_summaries` returns, so no plotting function knows the
difference. On the arsenic fit: **24.1 MB written in 0.5 s, read back in 1.7 s, against 24 s
to compute it again.**

All in `notebooks/moving_block_holdout_bootstrap.ipynb`, plus one line of `requirements.txt`.

### Why a table and not a bag of arrays

The interesting axis of this data *is* tabular — one row per reference combination — so the
file is a Parquet table with one row per combination and the bootstrap draws as list columns.
That makes the ranking every summary figure is built on readable by pandas, DuckDB, Polars or
R without unpacking a thousand draws per row, which a `.npz` of arrays cannot offer at any
size. Everything that is not per-combination — the design matrix, the energy grids, the
reference names, both clusterings, and the provenance — rides in the file's key-value
metadata as one JSON document with base64 arrays, which is where GeoParquet keeps its spec,
and leaves the table itself clean for anyone querying it.

Three decisions that account for the size:

- **The draws are float32.** They are 75 MB of the 83 MB at float64, and nothing drawn from
  them — medians, percentiles, violins — resolves anywhere near float32. The round trip is
  therefore exact for every other array and `rtol=1e-6` for these two, which the tests assert
  rather than assume.
- **The NaN padding is not written.** In memory `ref_indices`, `coef` and `bootstrap_coefs`
  are padded to `max_M`; Parquet list columns are variable-length, so each row stores its own
  `M` values and the reader puts the padding back. Nothing downstream can tell.
- **`fitted` is not written.** `fit_nnls` defines `residuals = fitted - b`, so it is exactly
  `b + residuals` — 3.7 MB of arithmetic.

Together with zstd that is 24.1 MB against the 82.6 MB the arrays occupy in memory, and well
under the 40 MB estimated when planning. Parquet's own column statistics say where it went:
`bootstrap_coefs` 15.70 MB, `bootstrap_pes` 6.43 MB, everything else under 1.5 MB together.

### What it stores, and what it does not

Coverage is deliberately the fit summaries — everything `do_fits_and_plot_summaries` draws.
Left out, each for a reason worth recording:

- **`holdout_masks` and `sampled_starts`**: only the holdout-geometry figures use them, and
  those take selector *callables* and redraw the masks themselves, so they could not be
  file-driven without refactoring.
- **`chance_merge_heights` and `cophenetic_distances`**: read only by
  `plot_cluster_metric_comparison`, and they would triple the metadata. The reader sets them
  to `None` rather than omitting the keys, so a caller sees why they are missing.
- **The references' raw measured points**: needed only by the interpolation-methods figure.
  Cheap at ~0.09 MB, and the obvious first addition if the studies are ever wanted from file.

`do_fits_and_plot_summaries` was widened to return `sample_energies`, `block_length`,
`n_holdout_blocks`, `elapsed_time` and the seed, which it computed and dropped. `block_length`
is the one that mattered: under `block_length='auto'` it is tuned from the residuals, so it is
a *result* of the fit that was being thrown away.

### The queries are the point

A section of six demonstrations, each answering a question without recomputing anything: the
provenance without reading a row; where the megabytes went; the best combinations from two
small columns; the spread of median prediction error by subset size; and one combination's
draws for an interval.

The fourth is the one worth keeping. **`Arsenopyrite_Julcani` and `orpiment` each appear in 21
of the best 25 three-component fits**, then a long tail — `As_pyrite` at 6, three others at 3.
The prediction-error ranking names one winner; this says two of its three references are
near-inevitable while the third is close to interchangeable. That is the same conclusion the
cosine tree suggested from a completely different direction, and it is a question the old
in-memory results could answer only by writing a loop.

### Notes

- 6 round-trip tests, on a miniature fit with real NaN padding and real clusterings, so they
  run in milliseconds: shapes and dtypes, exactness where it is promised and `rtol` where it
  is not, the padding, `fitted`, the provenance, the clusterings still drawing, the table
  ranking combinations the same way the draws do, and a foreign Parquet file being refused.
- `schema_version` is written into the file and checked on read, so a later format change
  fails loudly instead of mis-parsing.
- `pyarrow` is declared in `requirements.txt`, not just installed, so the writer and reader
  can move into the `mrfitty` package without a dependency change. Reading it back takes 1.7 s,
  nearly all of it Python-level list conversion rather than Arrow — the obvious thing to
  optimize if it ever matters.
- The file is a single row group, so column pruning saves I/O but row filters do not skip it.
  The query cell says so rather than implying a skip that is not happening.

---

## 2026-09-08 15:01 EDT — the cluster cutoff is coarse because the references are one family

The previous entry left a question open: `cluster_reference_spectra` cuts the tree at the
95th percentile of its randomized comparison, and on the arsenic pool that cut is
permissive — 22 of 23 merges fall below it, leaving two clusters and saying nothing about
the structure inside them. The suspicion was that the default percentile was wrong. It is
not, and this entry records what is wrong instead. No default changed.

All in `notebooks/moving_block_holdout_bootstrap.ipynb`.

### The percentile is not the lever

Sweeping it does almost nothing. From p50 to p99 the partition moves only between three
clusters and two, under both distances:

| percentile | correlation cutoff | clusters | cosine cutoff | clusters |
|---|---|---|---|---|
| p50 | 0.1624 | 3 | 0.0510 | 3 |
| p90 | 0.2881 | 2 | 0.0904 | 2 |
| **p95** | **0.3277** | **2** | **0.1034** | **2** |
| p99 | 0.3715 | 2 | 0.1179 | 2 |

The observed merges do not sit *inside* the distribution of randomized merge heights, they
sit far below all of it. A percentile of that distribution is therefore not a dial with
anything on the other end of it.

### One tempting fix is arithmetically a no-op

The obvious next idea was to preserve the shared absorption edge — model each reference as
the mean spectrum plus a deviation, and shuffle only the deviations. It returns
bit-identical cutoffs, because `mean + permute(A - mean)` and `permute(A)` are the same
multiset: permuting within a row is invariant to adding a row constant. Any comparison that
only re-centers rows is the comparison we already had.

### The prototype that was supposed to be harder is easier

`phase_randomize_columns` builds randomized references the standard surrogate-data way:
Fourier transform each reference, replace the phases with uniform random ones, transform
back, so each keeps its own power spectrum — mean, variance, smoothness — while whatever it
shares with the others is destroyed. `cluster_reference_spectra` grew a `surrogate=`
argument to select it, defaulting to `permute_within_rows` so nothing already measured
moved, and records which one produced a cutoff.

It is the weaker comparison, not the stronger one, and the reason is one number:

| | mean \|cross-reference correlation\| | spread across references at one energy |
|---|---|---|
| real references | 0.809 | 0.146 |
| `permute_within_rows` | **0.797** | 0.146 |
| `phase_randomize_columns` | **0.289** | 0.557 |

At any single energy the 24 references differ by very little, so shuffling which reference
holds which value leaves every randomized copy still tracing the same absorption edge — it
keeps 0.797 of the real 0.809. That makes the existing comparison a demanding one, with
small merge heights (median 0.162) that a real merge has to beat. Phase randomization throws
the shared edge away, its copies are nearly unrelated (0.289), its merge heights are large
(median 0.575, against the real tree's root of 0.571), and at p95 the cutoff lands above the
entire tree: **one cluster, every merge "significant", useless.**

### What this means for the cut

The two randomizations bracket the question rather than answering it — one preserves the
common edge, one destroys it, and neither leaves a percentile that carves the pool into
interpretable groups. That is a fact about the data: these references genuinely are
variations on one element's absorption edge. The coarse cut is not a defect to tune away,
and finer structure has to be read from merge heights, which is what the subtree brackets in
`plot_reference_dendrogram` already report.

`phase_randomize_columns` and the comparison stay in the notebook so the question can be
re-asked on a less homogeneous reference set, where it could come out differently.

### Wording

Standalone "null" is gone from the new section — headings, prose, the printed report and
`compare_surrogate_methods` (renamed from `compare_surrogate_nulls`) now say "randomized
copies", "comparison" or "randomization". The dict key `null_merge_heights` became
`chance_merge_heights`, all ten occurrences including the tests and
`plot_cluster_metric_comparison`. The word now appears once in the notebook, in the sentence
in `permute_within_rows` that defines it.

### Notes

- 14 clustering tests now: the surrogate must preserve each reference's power spectrum,
  mean, std and realness while lowering cross-reference correlation, and the `surrogate`
  argument must actually reach the cutoff rather than being accepted and ignored.
- Randomizing the DC term, or Nyquist when the length is even, would make the inverse
  transform complex; both phases are pinned to zero, which is also what keeps each
  reference's mean.
- Still unimplemented, and only worth doing if a descriptive partition is wanted: an
  `n_clusters` argument cutting with `criterion='maxclust'`, clearly not inferential.

---

## 2026-09-08 11:11 EDT — reference clustering, and the trees folded into the fit summaries

The notebook could say which combination of references fits best and how uncertain that
choice is, but nothing about how the chosen references relate to each other. That matters
for reading a result: NNLS cannot tell apart two references that are nearly the same
vector, so three references drawn from one tight cluster are a weaker claim about the
sample than three that are nothing alike, at identical prediction error. This batch adds
the clustering that answers it, a study settling which distance to use, and the wiring that
puts the answer next to each fit rather than in a section of its own.

All of it is `notebooks/moving_block_holdout_bootstrap.ipynb`; no package code changed.

### Clustering the design matrix

`cluster_reference_spectra(A, ref_names, rng, ...)` clusters the *columns of the design
matrix* rather than re-reading spectra from disk, so the tree describes exactly what was
fitted — same interpolation, same common energy range, same grid. It returns the linkage,
the pairwise distances, the flat clusters, the cophenetic correlation and the significance
cutoff in one dict, so nothing is recomputed at draw time.

The cutoff comes from a permutation test, reimplemented self-contained rather than reused
from `mrfitty/combination_fit.py`: `permute_within_rows` shuffles the values inside each
energy row, destroying resemblance between references while leaving each energy the values
it measured, and the cutoff is the 95th percentile of the merge heights those randomized
copies produce. It uses `rng.permuted(A, axis=1)`, which sidesteps the trap in the older
notebook copies — `df.values[i, :] = shuffle(...)` can silently no-op under pandas
copy-on-write, and the "null" then *is* the observed data, so the cutoff certifies whatever
it is given. Two tests pin that specifically.

`fcluster` and `cophenet` appear nowhere else in this repo; the prior art only ever drew
the cutoff line, never cut the tree with it.

### Showing where a combination sits

`plot_reference_dendrogram(clustering, highlight=...)` takes one combination or several as
`{label: refs}`, given as names or as column indices — so a NaN-padded row of
`results['ref_indices']` can be passed through unchanged. Per group it shades the smallest
subtree containing all of that combination's references and brackets it at the height that
subtree merges. That height is the whole point: near zero means the combination came out of
one cluster, at the root means it spans the reference set.

Mechanics worth recording, all verified against scipy rather than assumed:

- `orientation='left'` inverts the x axis (distance) and puts the leaves on y at
  `10 * position + 5`; the leaf labels sit *outside* the axes on the right, so highlight
  markers go inside via a blended transform.
- A subtree's leaves are always contiguous in drawn order, so the shading is one `axhspan`.
- scipy sizes the x axis from the root height alone, so a cutoff above the root would be
  drawn off-axes — the limits are recomputed from both before anything is added.
- The MRCA is found by unioning leaf sets and taking the smallest containing node, not by
  scanning heights, which would assume a monotonic linkage the `method` parameter permits
  callers to break.

### Which distance: correlation or cosine

A study in the style of the interpolation comparison, with a **Findings** cell. One design
matrix, one linkage method, the same randomized copies behind each cutoff — asserted, not
assumed, via a digest of the first permutation — and the same combinations located in both
trees.

The result is a split verdict. The two metrics agree almost perfectly on *neighbors*
(Spearman 0.991 over 276 pairs; one reference of 24 changes nearest neighbor) and disagree
on *where the tree is cut* (adjusted Rand 0.31): correlation isolates the five sulfides and
arsenides, cosine splits 14/10 along oxidation state, which is the chemically meaningful
line. It changes one reading — the best M=2 subset spans the root under correlation but
sits inside a single significant cluster under cosine, the difference between "two distinct
components" and "this pair may be partly interchangeable". Correlation stays the default
for continuity with the package and because its tree is the more faithful summary of its
own distances (cophenetic 0.901 vs 0.841); the recommendation is to read the cosine tree
alongside it whenever a selection falls below the cutoff.

### The trees now live with the fits

- `plot_ref_subsets_summary` draws both trees, unhighlighted, as its first figure — the
  reference pool as context for a section that covers every combination at once.
- `plot_bootstrap_summary` marks *that* combination in both trees.
- `do_fits_and_plot_summaries` clusters once per metric and passes a `{metric: clustering}`
  dict to both, so the 1,000-permutation test is not repeated on each of nine calls, and
  one place fixes the seed.
- `plot_reference_dendrogram` gained `legend_loc`, because its default legend hangs below
  the axes and lands on the next row when the panel is embedded.

**`plot_bootstrap_summary` is now one figure per row rather than one gridspec.** A shared
grid forces every row onto the same column edges, which split a two-panel row at
`n_cols // 2` — lopsided whenever `n_cols` is odd, so a two-reference fit got a 1:2 split
between its two trees — and left the per-reference panels narrow with wide gutters. Each
row now sizes its own panels; the rows share a width so they still read as one unit, and
the coefficient histograms and violins keep the same column count so a reference's violin
stays under its histogram.

### Order, and prose

The clustering section moved up to sit immediately after the reference interpolation that
builds the matrix it clusters; the highlighted demo and the metric study moved to just
after the fits — the earliest point where the selected subsets exist. Four cross-references
went stale in the move and were repaired, and the overview cell had been announcing "three
sections examine choices the pipeline makes" while listing three of what are now four.

Separately, every use of "null" as bare jargon was expanded for readers who do not already
know the method: the term is now defined once where the null model is built, and figures
say "tighter than 95% of merges from 1000 randomized copies" rather than "95th percentile
of the null merges".

### Notes

- 12 in-notebook tests, run by an explicit list rather than a `globals()` scan, which would
  re-run the earlier suites and misreport the count.
- The permutation test is cheap — 1,000 replicates in about 0.1 s at this size — so the
  clustering is not worth handing to `run_arms`.
- The cutoff is conservative on this data: 22 of 23 merges fall below it, leaving two
  clusters. It answers "is this grouping more than an accident", not "where does one group
  end"; finer structure has to be read off merge heights. Whether the default percentile
  should drop is still open.
- Verified by executing the notebook end to end (`jupyter nbconvert --execute`) after each
  change, not by spot checks; `nbstripout` already strips outputs at commit, so the
  figures live only in the working copy.

---

## 2026-09-07 12:25 EDT — pipeline overview at the top of the bootstrap notebook

`notebooks/moving_block_holdout_bootstrap.ipynb` opened straight into imports. Its
individual sections each carry a thorough write-up, but nothing said what the notebook as a
whole does, so the pipeline had to be reconstructed by reading 72 cells in order. Added one
markdown cell at the top that states it. Documentation only: no code, no figures and no
reported number changes.

### What the cell says

- **The question and the estimator.** Which combination of references explains the unknown
  spectrum, and how confident is that choice — both answered by the same moving-block
  holdout bootstrap.
- **The pipeline, in six steps**, each naming the function that performs it: read spectra →
  interpolate references onto the sample grid over the common energy range (`A`, `b`) →
  NNLS fit and residual diagnostics → pre-generate the shared holdout masks and moving-block
  starts → resample, refit, score prediction error at the held-out energies → sweep all
  2,324 reference combinations for M = 1, 2, 3 and rank by median prediction error.
- **The three studies** that follow, each with its conclusion in a line: the v1–v5 holdout
  geometry comparison, block-length tuning, and linear vs. cubic interpolation.

### Written for a reader who does not already know the method

The section write-ups below it assume the vocabulary; the overview does not, and this drove
most of the wording:

- A bootstrap is defined by what it does here — refit many versions of the same spectrum,
  each the fitted curve plus a reshuffled copy of the leftover noise, and take the spread of
  the resulting mixing coefficients and prediction errors as the measure of confidence.
- The autocorrelation of the residuals is described by its observable behavior — where the
  fitted curve runs above the measurement it stays above it across a stretch of adjacent
  energies, because the misfit is a smooth feature many points wide — which is also what
  makes a block the right unit to resample and to hold out. "Moving" is glossed as blocks
  that may start at any energy and may overlap.
- Why prediction error rather than goodness of fit is stated outright, since it is the
  premise the rest of the notebook rests on: adding a reference can only improve the fit, so
  residual size cannot say how many references are justified.

---

## 2026-09-06 22:32 EDT — `ssr` renamed to `rss_residuals`

Closes the item left open by the previous entry. `BootstrapValidationFitTask` called its
validation statistic `ssr` while computing `sqrt(sum(square(r)))` — the *root* sum of
squares of the validation residuals, their Euclidean norm, not the residual sum of squares
the abbreviation reads as. All three bootstrap methods compute it the same way, so the
name was consistently wrong rather than inconsistent, and the fix is a rename: no
arithmetic changed and no reported number moves.

### What was renamed

`ssr` → `rss_residuals` everywhere it named this quantity:

- the local variables at all three computation sites, and `ref_names_and_ssr`
- the `"ssr"` column of `bootstrap_df` and `bootstrap_coef_ci_df`
- the fit attributes `median_ssr`, `ssr_ci_lo` and `ssr_ci_hi`, and the two lookup dicts
  in `choose_best_component_count`
- the plot y-axis labels, the fit-quality line, and two log messages
- the six assertions in `test_bootstrap_validation_fit.py` that pin those names

`OlsWithStats.residual = self._result.ssr` in `linear_model.py` is untouched: that is
statsmodels' own attribute and a genuine residual sum of squares.

### What a user sees

Nothing in the numbers, and nothing in `write_table`'s TSV — that file carries `nss` and
`residuals_contribution`, never this column. Two strings in the PDF change ("Bootstrap RSS
residuals 95% ci of median" in the fit-quality block, "Bootstrap Validation RSS Residuals"
on the two boxplot axes), and the attribute names on `SpectrumFit` change, which is a break
for any downstream script reading `fit.median_ssr`.

### Incidentals

- `calculate_bootstrap_statistics`' docstring now defines the quantity once, explicitly, so
  the abbreviation cannot be misread again: *the root sum of squares of the validation
  residuals, `sqrt(sum(r ** 2))`, which is their Euclidean norm and not the residual sum of
  squares the abbreviation is sometimes used for.*
- Two docstrings in `plot.py` documented attributes that never existed
  (`median_ssr_ci_lo`, `median_ssr_ci_hi`); they now name the real ones.
- The six illustrative pandas tables in comments and docstrings were re-aligned, since
  `rss_residuals` is ten characters wider than `ssr` and the sample values no longer sat
  under their header. Two of those headers had been separating columns with a literal tab,
  which is why they rendered inconsistently; the file now has no tabs.

---

## 2026-09-06 22:10 EDT — three bugs behind eight failing tests

`pytest mrfitty/tests/` had eight failures. They looked like one problem — a matplotlib
API removal — and were three, two of which the first was hiding. All in `mrfitty/`;
nothing in the notebooks. The suite is back to 58 passed, 1 skipped.

### `matplotlib.cm.get_cmap` removed in 3.9

`plot_reference_tree` colored dendrogram leaf labels with `plt.cm.get_cmap("Accent", 2)`.
That call was deprecated in matplotlib 3.7 and removed in 3.9; the two-argument form's
replacement is `matplotlib.colormaps[name].resampled(n)`, which returns the same
two-entry lookup table, so `leaf_colors(0)` and `leaf_colors(1)` behave as before. The
commented-out variant a few lines above was updated to the same API so it still works if
uncommented. `plot.py` needed a plain `import matplotlib`, since `import
matplotlib.gridspec as gridspec` binds only `gridspec`.

This is why `requirements.txt` leaving `matplotlib` unpinned is now load-bearing: the fix
needs >= 3.6 for `colormaps` and `.resampled`.

### `"\n".join` over `Spectrum` objects

`AllCombinationFitTask.fit_all` catches every per-spectrum failure into `failed_fits` and
then reports them with `"\n".join(failed_fits)` — but that list holds `Spectrum` objects,
not names. Harmless until a fit actually fails, at which point it replaced the list of
what went wrong with `TypeError: sequence item 0: expected str instance, Spectrum found`.

That is what four of the eight tests were actually reporting: the colormap crash made
every fit fail, which tripped the join, which masked the real error. Now joins
`s.file_name`.

### Elementwise where a matrix product was meant

`calculate_bootstrap_statistics` predicted the validation half of the spectrum for every
bootstrap iteration with `A[valid_idx] * bootstrap_coefs`. `A[valid_idx]` is
(n_valid, n_refs) and `bootstrap_coefs` is (n_refs, bootstrap_count), so the result wants
to be (n_valid, bootstrap_count) — a matrix product, `@`.

**Why it survived.** With a single reference the shapes are (n_valid, 1) and
(1, bootstrap_count), and elementwise broadcasting produces *exactly* the array the matrix
product would. Checked `*` against `@` against an explicit per-iteration loop at
n_refs = 1, 2 and 3: at one reference all three agree, at two or more `*` raises
`ValueError: operands could not be broadcast together` and `@` matches the loop. So this
path had only ever run on one-component fits, which is also what the docstring's worked
example shows. No existing result changes; two- and three-component fits now work.

### `bootstrap_count` was ignored

With the `ValueError` gone, one test got far enough to fail on `assert 9999 == 100`.
`calculate_bootstrap_statistics` hardcoded `9999` for the resampled-residual draw while
the rest of the file honours `self.bootstrap_count` — which `fit_task_builder` plumbs
from `bootstrap_count` in the config's `[fit]` section, default 1000. Anyone who set that
option was silently getting 9999 iterations, roughly ten times the work they asked for.
Now uses `self.bootstrap_count`; the constructor default is 9999, so default behavior is
unchanged.

### Left open

`bootstrap_validation_ssr` is `sqrt(sum(square(...)))` — a residual norm, not the sum of
squared residuals its name and the docstring claim. Either the name is wrong or the
`sqrt` is. It feeds the "Bootstrap SSR 95% ci" line in the fit-quality text, so correcting
it would move published numbers; it needs a decision rather than a drive-by fix.

---

## 2026-09-06 21:24 EDT — the v1–v5 ranking was a statement about `n mod block_length`

Reworked the holdout-block sections of `notebooks/moving_block_holdout_bootstrap.ipynb`.
The block-structure figure was refactored into per-panel functions and redrawn at the
tuned block length, which exposed that the v1–v5 recommendation was an artifact of one
number dividing another. `select_holdout_blocks_v5` replaces v3 as the notebook's
selector, the affected sections were re-run, and the write-up's numbers are now all
produced by cells rather than quoted from working notes.

### Motivation

`plot_holdout_block_structure` was drawing its two rasters — the holdout masks and the
resample-block availability they cause — in non-adjacent rows, at different heights, on
different windows. Making them comparable meant extracting each panel type into its own
function, matching the raster geometry, and drawing the figure at `block_length=10`, the
value `choose_block_length` actually tunes for this data, rather than the rule-of-thumb 6.

At 10 the rasters no longer looked like the write-up described them.

### What the sweep found

`sweep_selector_block_lengths` runs every version at every block length from 4 to 20, at
three seeds each, and repeats the geometry at n ∈ {150, 198, 200, 233}.

**The original ranking was sampling noise.** A per-position holdout frequency is a mean of
B Bernoulli draws, so a perfectly uniform selector still measures a std across positions of
`sqrt(p(1-p)/B)` — 0.0149 at p = 1/3 and 1,000 iterations. At L = 6 the reported figures
were v1 0.0158, v3 0.0122, v4 0.0184, v5 0.0148: all four sit within a quarter of the
floor, and the spread across seeds 0–2 alone is [0.0148, 0.0158], [0.0122, 0.0151] and
[0.0129, 0.0184]. At 4,000 and 16,000 iterations all of them track the floor down as
1/√B. Only v2 (3.0× the floor, growing to 11.3× at 16,000 iterations) was ever measurable.

**What actually decides the ranking is whether the block length divides n.** A fixed-length
grid covers `floor(n/L) * L` positions and has to put the remainder somewhere:

- **v1** leaves it at the high-energy end permanently. At L = 10 positions 190–197 are held
  out in *zero* of 1,000 iterations. The count of never-held-out positions equals
  `n mod L` at every block length tested.
- **v3** wraps the grid modulo n, which does not create coverage it does not have — the
  uncovered arc is still `n mod L` wide, it just rotates with the offset, turning the dead
  zone into a ramp at each end. At L = 10 the frequency falls from 0.33 mid-spectrum to
  0.058 at position 0 and 0.068 at position 197.
- **v4/v5** have no remainder: they partition [0, n) into random-length blocks every
  iteration, so every position is in exactly one block at every block length.

v1 and v3 are at the sampling floor in exactly four of the seventeen block lengths swept —
6, 9, 11 and 18, the divisors of 198 in range — and 2.3–6.7× above it everywhere else.
v4 and v5 are within 1.1× of the floor at all seventeen, and across all 24 (n, L) pairs.
n = 200 is the control: there 10 divides n and 6 does not, and the versions swap places.

**This was live.** `do_ref_subsets_moving_block_holdout_bootstrap` defaults to
`block_length='auto'`, which tunes to 10 for this data, and every downstream cell ran v3 —
so the notebook's headline fits were using the one combination the sweep identifies as
worst. It is a knife-edge: the Politis–White p10 estimate is 10.28, and restricted to
subsets of M ≤ 2 it is 10.61, which rounds to 11 — a divisor of 198, where the defect
vanishes entirely. Uniformity should not depend on whether a tuned estimate happens to
round onto a divisor of the number of energy points.

Prediction error does not discriminate either way: every version's mean-PE CI overlaps
every other's at L = 6, 10 and 15, as the original analysis said.

### What changed

- **`plot_holdout_block_structure`** — nine panel functions extracted (one per row), rows
  reordered so the two rasters are adjacent, and the resample-start rasters matched to the
  mask rasters in height (`START_RASTER_HEIGHT_FRACTION`, so the caller sizes the row
  rather than hardcoding a magic ratio), in window, and in the aligned grid ruled across
  them. Runs at `block_length=10`.
- **New sweep cells** — the machinery, a four-panel sensitivity figure, the sweep itself,
  a sampling-floor convergence check, and the structural geometry and ACF tables.
- **v3 → v5 everywhere downstream** — `make_block_length_selector`'s default,
  `do_fits_and_plot_summaries`, `compare_interpolation_methods`. The v1–v5 pipeline
  comparison now runs at both 6 and 10, since the point is how the ranking moves with L.
- **Re-ran the affected sections.** The interpolation comparison is unchanged in substance:
  ρ 0.99995 → 0.99994, the same subset selected at every size, cubic still marginally
  better at the optimum. The block-length study's null result is *stronger* under v5 — the
  selected subset is identical from L = 3 to 15 and the rank correlation against L = 6 never
  drops below 0.987, against 0.969 under v3, suggesting some of the earlier drift with
  block length was v3's own coverage defect growing with the remainder. New finding: at
  L = 20 the selection does move, at all three subset sizes.

### Reproducibility and runtime

Every number the v1–v5 write-up quotes is now printed by a cell — 125 of 125 numeric
tokens, checked mechanically against the stored outputs. That audit turned up three
errors of its own:

- The wrap-rate prediction was wrong. Holdout blocks are drawn *without* replacement, so
  the closed forms are (5/6)(11/33) + (1/6)(11/33)(10/32) = **29.5%** for v3 and
  (11/33)(10/32) = **10.4%** for v1, not the 27.8% and 11.1% the write-up gave. The old
  numbers looked right only because a rate near 0.3 carries a 1.4-point standard error at
  1,000 iterations. `predicted_both_ends_held_out` derives them for any (n, L), and the
  cell re-measures at 40,000 iterations, where both land: 29.3% and 10.5%.
- The L = 6 results table's confidence-interval columns had drifted from what the cell
  prints.
- v5's std is 1.00× the floor, not 0.99×.

The sweeps and the four heaviest cells now run their arms through `joblib.Parallel`
(loky backend, `N_JOBS` at the top of the sweep machinery). Every arm already built its
own `default_rng(seed)` and shares nothing, so this is a scheduling change and not a
numerical one — the re-run's outputs are byte-identical for the cells that were only
parallelized, and the sweep cell asserts it every run by re-executing one arm in-process
and comparing element-wise. The notebook went from 5m18s to 2m02s.

### Notes

- loky rather than a forked `ProcessPoolExecutor`: both measured the same speedup and both
  gave identical results, but loky pickles the notebook's functions by value, so it does
  not depend on workers inheriting `__main__` and does not fork a process that has already
  loaded matplotlib and BLAS. joblib arrives with `scikit-learn`, already in
  `requirements.txt`.
- Arms run in worker processes, so they take everything they need as arguments — including
  the selector — and hand back whatever they printed, since a worker's stdout never reaches
  the notebook. `run_arms` returns results in task order so the printed tables do not
  depend on scheduling.
- v4/v5's uniformity is not an artifact of their holding out slightly more data. An
  explicit `block_length=L` slides their random range to [L, L+4], so at nominal L they
  hold out ~4% more of the spectrum in longer blocks; the sweep reports realized run length
  and held-out fraction alongside. They sit at the floor at every L from 4 to 20, while
  v1/v3 sit at it only at divisors.
- The "v4/v5 leave a larger resample pool" finding from the earlier structural write-up is
  retired: it is +5 points at L = 6, gone at L = 10, and reversed by L = 15.

---

## 2026-07-20 21:03 EDT — auto-generated table of contents for this journal

Added a `## Contents` section to the top of this file and made it self-maintaining, so
new entries no longer need a hand-written link.

### How it works

- **`scripts/gen_toc.py`** — a stdlib-only generator. It reads the `## ` entry headings,
  builds GitHub-style anchor slugs (lowercase; backticks, colons and commas dropped;
  ` — ` collapses to `--`), and rewrites whatever sits between the `<!-- toc -->` and
  `<!-- /toc -->` markers. It skips fenced code blocks and the `## Contents` heading
  itself, and de-duplicates slugs the way GitHub does. Run directly as
  `python3 scripts/gen_toc.py dev-journal.md`.
- **`.pre-commit-config.yaml`** — a `local` hook, `journal-toc`, runs the script whenever
  a commit touches `dev-journal.md`. Like the `black` hook already here, it rewrites the
  file and exits non-zero if it changed anything, so the commit stops and the refreshed
  TOC gets re-staged.

### Notes

- `language: system` with `python3`, matching the config's `default_language_version`.
  The script has no third-party imports, so it does not need the `mrfitty-py313` env.
- The hook fires only on commit, not on editor or Jupyter saves. To preview the TOC
  before committing, run the script by hand.
- Verified end to end: a no-change run passes; injecting a new heading makes the hook
  repair the TOC and fail once, then pass on re-run, with correctly slugged anchors.

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
