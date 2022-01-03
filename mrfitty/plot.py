import datetime
import itertools
import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hc


def plot_fit(spectrum, any_given_fit, title, fit_quality_labels):
    log = logging.getLogger(name=__name__ + ":" + spectrum.file_name)

    f, ax = plt.subplots()
    # ax.set_title(spectrum.file_name + '\n' + title + '\n' + self.get_fit_quality_score_text(any_given_fit))
    ax.set_title(title + "\n" + spectrum.file_name)
    # log.info(any_given_fit.fit_spectrum_b.shape)

    reference_contributions_percent_sr = any_given_fit.get_reference_contributions_sr()
    reference_only_contributions_percent_sr = (
        any_given_fit.get_reference_only_contributions_sr()
    )
    longest_name_len = max(
        [len(name) for name in reference_contributions_percent_sr.index]
        + [len(spectrum.file_name)]
    )
    # the format string should look like '{:N}{:5.2f} ({:5.2f})' where N is the length
    #   of the longest reference name
    reference_contribution_format_str = (
        "{:" + str(longest_name_len + 4) + "}{:5.2f} ({:5.2f})"
    )
    residuals_contribution_format_str = "{:" + str(longest_name_len + 4) + "}{:5.2f}"

    # add fits in descending order of reference contribution
    reference_line_list = []
    reference_label_list = []
    reference_contributions_percent_sr.sort_values(ascending=False, inplace=True)
    reference_only_contributions_percent_sr.sort_values(ascending=False, inplace=True)
    log.debug("plotting reference components")
    log.debug(reference_contributions_percent_sr.head())
    for (ref_name, ref_contrib), (ref_only_name, ref_only_contrib) in zip(
        reference_contributions_percent_sr.items(),
        reference_only_contributions_percent_sr.items(),
    ):
        log.debug("reference contribution %s %5.2f", ref_name, ref_contrib)
        log.debug(
            "reference-only contribution %s %5.2f", ref_only_name, ref_only_contrib
        )
        reference_label = reference_contribution_format_str.format(
            ref_name, ref_contrib, ref_only_contrib
        )
        reference_label_list.append(reference_label)

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
            *reference_label_list,
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


def plot_stacked_fit(spectrum, any_given_fit, title, fit_quality_labels):
    log = logging.getLogger(name=__name__)

    f, ax = plt.subplots()
    ax.set_title(title + "\n" + spectrum.file_name)
    # log.info(any_given_fit.fit_spectrum_b.shape)

    reference_contributions_percent_sr = any_given_fit.get_reference_contributions_sr()
    longest_name_len = max(
        [len(name) for name in reference_contributions_percent_sr.index]
        + [len(spectrum.file_name)]
    )
    # the format string should look like '{:N}{:5.2f}' where N is the length of the longest reference name
    contribution_format_str = "{:" + str(longest_name_len + 4) + "}{:5.2f}"

    # log.info(any_given_fit.residuals.shape)
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

    # add fits in descending order of reference contribution
    reference_label_list = []
    reference_contributions_percent_sr.sort_values(ascending=False, inplace=True)
    sort_ndx = reversed(any_given_fit.reference_spectra_coef_x.argsort())
    ys = any_given_fit.reference_spectra_coef_x * any_given_fit.reference_spectra_A_df
    log.debug("plotting reference components")
    log.debug(reference_contributions_percent_sr.head())
    reference_contributions_percent_sr.sort_values(ascending=False)
    for name, value in reference_contributions_percent_sr.items():
        log.debug("reference component {} {}".format(name, value))
        reference_label = contribution_format_str.format(name, value)
        reference_label_list.append(reference_label)

    reference_line_list = ax.stackplot(
        ys.index, *[ys.iloc[:, i] for i in sort_ndx], labels=reference_label_list
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
            *reference_label_list,
            residuals_label,
            *fit_quality_labels,
        ],
        prop=dict(family="Monospace", size=7),
    )

    add_date_time_footer(ax)

    plt.tight_layout()

    return f


def add_date_time_footer(ax):
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
    z_star = 1.96
    Z = z_star * (np.std(C_p_list) / np.sqrt(len(C_p_list)))
    mean_C_p = np.mean(C_p_list)
    lo = mean_C_p - Z
    hi = mean_C_p + Z
    return lo, hi


def box_and_histogram(C_p_list, ax_box, ax_hist, hist_ylim=None):
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
