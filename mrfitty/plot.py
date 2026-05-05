import datetime
import itertools
import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hc


def plot_fit(spectrum, any_given_fit, title, fit_quality_labels, reference_to_reference_label):
    """Plot a spectrum fit with reference contributions, residuals, and fit quality labels.

    Creates a single-axes figure showing the unknown spectrum, the combined fit line,
    per-reference weighted contributions (listed in descending order), and residuals.

    Parameters
    ----------
    spectrum : Spectrum
        The unknown spectrum being fitted; used for its file_name label.
    any_given_fit : SpectrumFit
        Fit result object providing interpolant_incident_energy, fit_spectrum_b,
        unknown_spectrum_b, and residuals / residuals_contribution.
    title : str
        Primary title line shown above the spectrum file name.
    fit_quality_labels : list of str
        Additional annotation strings appended to the legend (e.g. NSS score text).
    reference_to_reference_label : dict of {str: str}
        Mapping from reference name to formatted legend label string, in the order
        they should appear in the legend.  Built by the calling FitTask via
        ``build_reference_to_reference_label``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure ready for saving or display.
    """
    log = logging.getLogger(name=__name__ + ":" + spectrum.file_name)

    f, ax = plt.subplots()
    # ax.set_title(spectrum.file_name + '\n' + title + '\n' + self.get_fit_quality_score_text(any_given_fit))
    ax.set_title(title + "\n" + spectrum.file_name)

    pad = max(max(len(k) for k in reference_to_reference_label), len(spectrum.file_name)) + 4
    residuals_contribution_format_str = "{:" + str(pad) + "}{:5.2f}"

    # add fits in descending order of reference contribution
    reference_line_list = []
    log.debug("plotting reference components")
    for ref_name, reference_label in reference_to_reference_label.items():
        log.debug("plotting reference component %s", ref_name)
        # plot once for each reference just to build the legend
        # ax.plot returns a list
        reference_line_list.extend(
            ax.plot(
                any_given_fit.interpolant_incident_energy,
                any_given_fit.fit_spectrum_b,
                label=reference_label,
                color="w",
                alpha=0.0,
            )
        )

    # log.info(any_given_fit.residuals.shape)
    residuals_label = residuals_contribution_format_str.format(
        "residuals", any_given_fit.residuals_contribution
    )
    residuals_line = ax.plot(
        any_given_fit.interpolant_incident_energy,
        any_given_fit.residuals,
        ".",
        label=residuals_label,
        alpha=0.5,
    )

    fit_line_label = "fit"
    fit_line = ax.plot(
        any_given_fit.interpolant_incident_energy,
        any_given_fit.fit_spectrum_b,
        label=fit_line_label,
    )

    spectrum_points = ax.plot(
        any_given_fit.interpolant_incident_energy,
        any_given_fit.unknown_spectrum_b,
        ".",
        label=spectrum.file_name,
        alpha=0.5,
    )

    # add some fake lines to create some special legend entries
    fit_quality_lines = []
    for fit_quality_label in fit_quality_labels:
        fit_quality_lines.extend(
            ax.plot([], [], label=fit_quality_label, color="w", alpha=0.0)
        )

    ax.set_xlabel("eV")
    ax.set_ylabel("normalized absorbance")
    # TODO: make these configurable
    legend_location = "best"
    legend_font_size = 6
    ax.legend(
        [
            *reference_line_list,
            *spectrum_points,
            *residuals_line,
            *fit_line,
            *fit_quality_lines,
        ],
        [
            *reference_to_reference_label.values(),
            spectrum.file_name,
            residuals_label,
            fit_line_label,
            *fit_quality_labels,
        ],
        loc=legend_location,
        prop=dict(family="Monospace", size=legend_font_size),
    )

    # ax.text(
    #     0.75,
    #     0.35,
    #     self.get_fit_quality_score_text(any_given_fit=any_given_fit),
    #     transform=ax.transAxes,
    #     fontsize=6,
    #     verticalalignment='top',
    #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # )

    add_date_time_footer(ax)

    plt.tight_layout()

    return f


def plot_stacked_fit(spectrum, any_given_fit, title, fit_quality_labels, reference_to_reference_label):
    """Plot reference contributions as a stacked area chart with the unknown spectrum overlaid.

    Each reference component is scaled by its fitted coefficient and stacked on top of
    the previous ones, giving a visual breakdown of how each reference builds up toward
    the total fit.  The unknown spectrum and residuals are drawn as scatter points on
    top.

    Parameters
    ----------
    spectrum : Spectrum
        The unknown spectrum being fitted; used for its file_name label.
    any_given_fit : SpectrumFit
        Fit result object providing interpolant_incident_energy, unknown_spectrum_b,
        residuals, residuals_contribution, reference_spectra_coef_x, and
        reference_spectra_A_df.
    title : str
        Primary title line shown above the spectrum file name.
    fit_quality_labels : list of str
        Additional annotation strings appended to the legend (e.g. NSS score text).
    reference_to_reference_label : dict of {str: str}
        Mapping from reference name to formatted legend label string, in the order
        they should appear in the legend.  Built by the calling FitTask via
        ``build_stacked_reference_to_reference_label``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure ready for saving or display.
    """
    log = logging.getLogger(name=__name__)

    f, ax = plt.subplots()
    ax.set_title(title + "\n" + spectrum.file_name)

    pad = max(max(len(k) for k in reference_to_reference_label), len(spectrum.file_name)) + 4
    contribution_format_str = "{:" + str(pad) + "}{:5.2f}"

    residuals_label = contribution_format_str.format(
        "residuals", any_given_fit.residuals_contribution
    )
    residuals_line = ax.plot(
        any_given_fit.interpolant_incident_energy,
        any_given_fit.residuals,
        ".",
        label=residuals_label,
        alpha=0.5,
    )

    spectrum_points = ax.plot(
        any_given_fit.interpolant_incident_energy,
        any_given_fit.unknown_spectrum_b,
        ".",
        label=spectrum.file_name,
        alpha=0.5,
    )

    sort_ndx = reversed(any_given_fit.reference_spectra_coef_x.argsort())
    ys = any_given_fit.reference_spectra_coef_x * any_given_fit.reference_spectra_A_df
    log.debug("plotting reference components")
    reference_line_list = ax.stackplot(
        ys.index, *[ys.iloc[:, i] for i in sort_ndx], labels=list(reference_to_reference_label.values())
    )

    # add some fake lines to create some special legend entries
    fit_quality_lines = []
    for fit_quality_label in fit_quality_labels:
        fit_quality_lines.extend(
            ax.plot([], [], label=fit_quality_label, color="w", alpha=0.0)
        )

    ax.set_xlabel("eV")
    ax.set_ylabel("normalized absorbance")
    ax.legend(
        # these arguments are documented but this does not seem to work
        [*spectrum_points, *reference_line_list, *residuals_line, *fit_quality_lines],
        [
            spectrum.file_name,
            *reference_to_reference_label.values(),
            residuals_label,
            *fit_quality_labels,
        ],
        prop=dict(family="Monospace", size=7),
    )

    add_date_time_footer(ax)

    plt.tight_layout()

    return f


def add_date_time_footer(ax):
    """Annotate the figure with the current ISO-format timestamp in the lower-left corner.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Any axes belonging to the target figure; the annotation is placed in figure
        fraction coordinates so the specific axes does not matter.

    Returns
    -------
    None
    """
    ax.annotate(
        datetime.datetime.now().isoformat(),
        xy=(0.025, 0.025),
        xycoords="figure fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=4,
    )


def plot_reference_tree(
    linkage_distance_variable_by_sample,
    reference_df,
    cutoff_distance,
    title,
    reference_spectra_names,
    pdist_metric,
    linkage_method,
):
    """Plot a hierarchical clustering dendrogram of the reference spectra.

    Draws a horizontal dendrogram with a vertical line at the cutoff distance.
    Leaf labels that belong to the active reference set are coloured differently
    from the remaining reference spectra to make selection visible at a glance.

    Parameters
    ----------
    linkage_distance_variable_by_sample : ndarray
        Linkage matrix as returned by ``scipy.cluster.hierarchy.linkage``.
    reference_df : pandas.DataFrame
        DataFrame whose columns are reference spectrum names; used to label
        the dendrogram leaves.
    cutoff_distance : float
        Distance threshold drawn as a vertical line on the dendrogram.
    title : str
        Title text placed above the plot.
    reference_spectra_names : iterable of str
        Names of the reference spectra included in the current fit; their leaf
        labels are highlighted with a distinct colour.
    pdist_metric : str
        Pairwise-distance metric name (e.g. ``"euclidean"``); shown on the x-axis.
    linkage_method : str
        Linkage method name (e.g. ``"ward"``); shown on the x-axis.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure ready for saving or display.
    """
    # log = logging.getLogger(__name__)

    f, ax = plt.subplots()
    dendrogram = hc.dendrogram(
        ax=ax,
        Z=linkage_distance_variable_by_sample,
        orientation="left",
        leaf_font_size=8,
        labels=reference_df.columns,
    )

    # leaf_colors = plt.cm.get_cmap("Accent", 2)
    # for i, leaf_label in enumerate(plt.gca().get_ymajorticklabels()):
    #    leaf_label.set_color(leaf_colors(i % 2))

    icoords = tuple([i for i in itertools.chain(dendrogram["icoord"])])
    ax.vlines(cutoff_distance, ymin=np.min(icoords), ymax=np.max(icoords))
    ax.set_title("{}\n".format(title))
    ax.set_xlabel("{} distance\n{} linkage".format(pdist_metric, linkage_method))

    leaf_colors = plt.cm.get_cmap("Accent", 2)
    for leaf_label in plt.gca().get_ymajorticklabels():
        if leaf_label.get_text() in reference_spectra_names:
            leaf_label.set_color(leaf_colors(1))
        else:
            leaf_label.set_color(leaf_colors(0))

    add_date_time_footer(ax)
    plt.tight_layout()

    return f


def plot_prediction_errors(spectrum, fit, title):
    """Plot boxplots and histograms of prediction errors at three bootstrap sample sizes.

    Produces a single figure with six sub-axes arranged in pairs (boxplot + histogram)
    for the first 10, 100, and 1 000 bootstrap prediction-error samples.  The histogram
    y-axis limits are shared so distributions are directly comparable.

    Parameters
    ----------
    spectrum : Spectrum
        The unknown spectrum (currently unused inside the function, reserved for the
        figure title in future revisions).
    fit : SpectrumFit
        Fit result providing a ``prediction_errors`` sequence of bootstrap prediction
        error values.
    title : str
        Text placed in the figure suptitle above "Prediction Errors".

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure ready for saving or display.
    """
    f = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=1, ncols=6)
    gs.update(wspace=1.0)
    f.suptitle(title + "\n" + "Prediction Errors")

    ax_box_1000 = plt.subplot(gs[0, 4])
    ax_hist_1000 = plt.subplot(gs[0, 5])
    box_and_histogram(
        fit.prediction_errors[:1000], ax_box=ax_box_1000, ax_hist=ax_hist_1000
    )

    ax_box_100 = plt.subplot(gs[0, 2])
    ax_hist_100 = plt.subplot(gs[0, 3])
    box_and_histogram(
        fit.prediction_errors[:100],
        hist_ylim=ax_hist_1000.get_ylim(),
        ax_box=ax_box_100,
        ax_hist=ax_hist_100,
    )
    ax_box_10 = plt.subplot(gs[0, 0])
    ax_hist_10 = plt.subplot(gs[0, 1])
    box_and_histogram(
        fit.prediction_errors[:10],
        hist_ylim=ax_hist_1000.get_ylim(),
        ax_box=ax_box_10,
        ax_hist=ax_hist_10,
    )

    add_date_time_footer(ax_box_1000)
    plt.tight_layout()

    return f


def classical_confidence_interval_of_mean(C_p_list):
    """Compute a 95 % confidence interval for the mean using the z-distribution.

    Uses the formula ``mean ± 1.96 * (std / sqrt(n))``, which assumes the
    sample is large enough for the central-limit theorem to apply.

    Parameters
    ----------
    C_p_list : array-like of float
        Sample of prediction-error (C_p) values.

    Returns
    -------
    lo : float
        Lower bound of the 95 % confidence interval.
    hi : float
        Upper bound of the 95 % confidence interval.
    """
    z_star = 1.96
    Z = z_star * (np.std(C_p_list) / np.sqrt(len(C_p_list)))
    mean_C_p = np.mean(C_p_list)
    lo = mean_C_p - Z
    hi = mean_C_p + Z
    return lo, hi


def box_and_histogram(C_p_list, ax_box, ax_hist, hist_ylim=None):
    """Draw a notched boxplot and a horizontal histogram for a list of prediction errors.

    Parameters
    ----------
    C_p_list : array-like of float
        Sample of prediction-error (C_p) values to visualise.
    ax_box : matplotlib.axes.Axes
        Axes on which the notched boxplot is drawn.
    ax_hist : matplotlib.axes.Axes
        Axes on which the horizontal histogram is drawn; its x-axis label shows
        the sample size ``N``.
    hist_ylim : tuple of (float, float) or None, optional
        If provided, both axes are constrained to this y-axis range so that
        multiple ``box_and_histogram`` panels are directly comparable.

    Returns
    -------
    ax_box : matplotlib.axes.Axes
        The boxplot axes after drawing.
    ax_hist : matplotlib.axes.Axes
        The histogram axes after drawing.
    """
    if hist_ylim is None:
        pass
    else:
        ax_box.set_ylim(hist_ylim)
        ax_hist.set_ylim(hist_ylim)

    ax_box.boxplot(C_p_list, notch=True)
    # ax_box.scatter((1, 1), (ci_mean_lo, ci_mean_hi))

    ax_hist.hist(C_p_list, orientation="horizontal", bins=20)
    ax_hist.set_xlabel(f"N={len(C_p_list)}")

    return ax_box, ax_hist


def prediction_error_box_plots(ax, title, sorted_fits):
    """Draw notched boxplots of prediction-error distributions for the top fits.

    Each box corresponds to one fit in ``sorted_fits``, ordered left to right.
    User-supplied medians and confidence intervals from the fit objects override
    the computed medians.  NSS values are overlaid as ``x`` scatter markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the boxplots are drawn.
    title : str
        Title placed above the axes.
    sorted_fits : list of SpectrumFit
        Fit result objects providing ``prediction_errors``, ``median_C_p``,
        ``median_C_p_ci_lo``, ``median_C_p_ci_hi``, and ``nss``.

    Returns
    -------
    None
    """
    ax.boxplot(
        x=[fit_i.prediction_errors for fit_i in sorted_fits],
        usermedians=[fit_i.median_C_p for fit_i in sorted_fits],
        conf_intervals=[
            [fit_i.median_C_p_ci_lo, fit_i.median_C_p_ci_hi] for fit_i in sorted_fits
        ],
        notch=True,
    )
    ax.scatter(
        x=range(1, len(sorted_fits) + 1),
        y=[fit_i.nss for fit_i in sorted_fits],
        marker="x",
    )

    ax.set_title(title)
    ax.set_xlabel(f"top {len(sorted_fits)} fits")
    ax.set_ylabel("Prediction Error")

    add_date_time_footer(ax)


def prediction_error_confidence_interval_plot(ax, title, sorted_fits):
    """Plot median prediction-error confidence intervals for the top fits as error bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the error-bar plot is drawn.
    title : str
        Title placed above the axes.
    sorted_fits : list of SpectrumFit
        Fit result objects providing ``median_C_p``, ``median_C_p_ci_lo``, and
        ``median_C_p_ci_hi``.

    Returns
    -------
    None
    """
    ax.errorbar(
        y=[spectrum_fit.median_C_p for spectrum_fit in sorted_fits],
        x=range(len(sorted_fits)),
        yerr=[
            [s.median_C_p - s.median_C_p_ci_lo for s in sorted_fits],
            [s.median_C_p_ci_hi - s.median_C_p for s in sorted_fits],
        ],
        fmt="o",
    )
    ax.set_title(title)
    ax.set_xlabel(f"top {len(sorted_fits)} fits")
    ax.set_ylabel("Prediction Error Confidence Intervals")
    ax.grid()

    add_date_time_footer(ax)


def best_fit_for_component_count_box_plots(ax, title, top_fit_per_component_count):
    """Draw notched boxplots of prediction errors for the best fit at each component count.

    One box is drawn per component count, ordered by component count on the x-axis.
    User-supplied medians and confidence intervals from the fit objects override the
    computed medians.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the boxplots are drawn.
    title : str
        Title placed above the axes.
    top_fit_per_component_count : dict of {int: SpectrumFit}
        Mapping from component count to the best fit result for that count.  Each
        fit provides ``prediction_errors``, ``median_C_p``, ``median_C_p_ci_lo``,
        and ``median_C_p_ci_hi``.

    Returns
    -------
    None
    """
    ax.boxplot(
        x=[
            fit_i.prediction_errors
            for i, fit_i in sorted(top_fit_per_component_count.items())
        ],
        usermedians=[
            fit_i.median_C_p for i, fit_i in sorted(top_fit_per_component_count.items())
        ],
        conf_intervals=[
            [fit_i.median_C_p_ci_lo, fit_i.median_C_p_ci_hi]
            for i, fit_i in sorted(top_fit_per_component_count.items())
        ],
        notch=True,
    )
    ax.set_title(title)
    ax.set_xlabel("component count")
    ax.set_ylabel("Prediction Error")

    add_date_time_footer(ax)


def bootstrap_validation_box_plots(ax, title, sorted_fits):
    """Draw notched boxplots of bootstrap SSR distributions for the top fits.

    Each box corresponds to one fit in ``sorted_fits``, ordered left to right.
    User-supplied medians and confidence intervals from the fit objects override
    the computed medians.  NSS values are overlaid as ``x`` scatter markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the boxplots are drawn.
    title : str
        Title placed above the axes.
    sorted_fits : list of BootstrapValidationFit
        Fit result objects providing ``bootstrap_df`` (with an ``"ssr"`` column),
        ``median_ssr``, ``median_ssr_ci_lo``, ``median_ssr_ci_hi``, and ``nss``.

    Returns
    -------
    None
    """
    ax.boxplot(
        x=[fit_i.bootstrap_df["ssr"] for fit_i in sorted_fits],
        usermedians=[fit_i.median_ssr for fit_i in sorted_fits],
        conf_intervals=[
            [fit_i.median_ssr_ci_lo, fit_i.median_ssr_ci_hi] for fit_i in sorted_fits
        ],
        notch=True,
    )
    ax.scatter(
        x=range(1, len(sorted_fits) + 1),
        y=[fit_i.nss for fit_i in sorted_fits],
        marker="x",
    )
    ax.set_title(title)
    ax.set_xlabel(f"top {len(sorted_fits)} fits")
    ax.set_ylabel("Bootstrap Validation SSR")
    add_date_time_footer(ax)


def bootstrap_validation_confidence_interval_plot(ax, title, sorted_fits):
    """Plot median bootstrap SSR confidence intervals for the top fits as error bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the error-bar plot is drawn.
    title : str
        Title placed above the axes.
    sorted_fits : list of BootstrapValidationFit
        Fit result objects providing ``median_ssr``, ``median_ssr_ci_lo``, and
        ``median_ssr_ci_hi``.

    Returns
    -------
    None
    """
    ax.errorbar(
        y=[fit.median_ssr for fit in sorted_fits],
        x=range(len(sorted_fits)),
        yerr=[
            [s.median_ssr - s.median_ssr_ci_lo for s in sorted_fits],
            [s.median_ssr_ci_hi - s.median_ssr for s in sorted_fits],
        ],
        fmt="o",
    )
    ax.set_title(title)
    ax.set_xlabel(f"top {len(sorted_fits)} fits")
    ax.set_ylabel("Bootstrap Validation SSR Confidence Intervals")
    ax.grid()
    add_date_time_footer(ax)


def best_bootstrap_fit_for_component_count_box_plots(ax, title, top_fit_per_component_count):
    """Draw notched boxplots of bootstrap SSR for the best fit at each component count.

    One box is drawn per component count, ordered by component count on the x-axis.
    User-supplied medians and confidence intervals from the fit objects override the
    computed medians.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the boxplots are drawn.
    title : str
        Title placed above the axes.
    top_fit_per_component_count : dict of {int: BootstrapValidationFit}
        Mapping from component count to the best fit result for that count.  Each
        fit provides ``bootstrap_df`` (with an ``"ssr"`` column), ``median_ssr``,
        ``median_ssr_ci_lo``, and ``median_ssr_ci_hi``.

    Returns
    -------
    None
    """
    ax.boxplot(
        x=[
            fit_i.bootstrap_df["ssr"]
            for i, fit_i in sorted(top_fit_per_component_count.items())
        ],
        usermedians=[
            fit_i.median_ssr for i, fit_i in sorted(top_fit_per_component_count.items())
        ],
        conf_intervals=[
            [fit_i.median_ssr_ci_lo, fit_i.median_ssr_ci_hi]
            for i, fit_i in sorted(top_fit_per_component_count.items())
        ],
        notch=True,
    )
    ax.set_title(title)
    ax.set_xlabel("component count")
    ax.set_ylabel("Bootstrap Validation SSR")
    add_date_time_footer(ax)
