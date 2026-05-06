from collections import defaultdict
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikits.bootstrap

from mrfitty.base import AdaptiveEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.linear_model import NonNegativeLinearRegression
from mrfitty.plot import (
    bootstrap_validation_box_plots,
    bootstrap_validation_confidence_interval_plot,
    best_bootstrap_fit_for_component_count_box_plots,
)
from mrfitty.prediction_error_fit import PredictionErrorFitTask


class BootstrapValidationFitTask(AllCombinationFitTask):
    def __init__(
        self,
        reference_spectrum_list,
        unknown_spectrum_list,
        ls=NonNegativeLinearRegression,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        component_count_range=range(1, 4),
        best_fits_plot_limit=3,
        bootstrap_count=1000,
    ):
        super().__init__(
            ls=ls,
            reference_spectrum_list=reference_spectrum_list,
            unknown_spectrum_list=unknown_spectrum_list,
            energy_range_builder=energy_range_builder,
            best_fits_plot_limit=best_fits_plot_limit,
            component_count_range=component_count_range,
        )
        self.bootstrap_count = bootstrap_count

    def get_fit_quality_score_text(self, any_given_fit):
        return [
            "Bootstrap SSR 95% ci of median: {:.3f} <-- {:.3f} --> {:.3f}".format(
                any_given_fit.median_ssr_ci_lo,
                any_given_fit.median_ssr,
                any_given_fit.median_ssr_ci_hi,
            ),
            "MSE: {:<8.3f}".format(any_given_fit.nss),
        ]

    def calculate_bootstrap_results(self, fit):
        """
        Split the spectrum into even-indexed (training) and odd-indexed (validation) data points.
        Run bootstrap_count bootstrapped NNLS fits on samples drawn with replacement from the
        training set. For each fit record the reference coefficients and the SSR on the
        validation set.

        Parameters
        ----------
        fit : SpectrumFit

        Returns
        -------
        pd.DataFrame
            Rows = bootstrap iterations, columns = [ref1_name, ..., refN_name, "ssr"].
        """
        n = len(fit.unknown_spectrum_b)
        train_idx = np.arange(0, n, 2)
        valid_idx = np.arange(1, n, 2)

        A = fit.reference_spectra_A_df.values
        b = fit.unknown_spectrum_b.values
        ref_names = list(fit.reference_spectra_A_df.columns)

        records = []
        for _ in range(self.bootstrap_count):
            boot_idx = np.random.choice(train_idx, size=len(train_idx), replace=True)
            lm = self.ls()
            lm.fit(A[boot_idx], b[boot_idx])
            predicted = lm.predict(A[valid_idx])
            ssr = float(np.sqrt(np.sum((b[valid_idx] - predicted) ** 2)))
            record = dict(zip(ref_names, lm.coef_))
            record["ssr"] = ssr
            records.append(record)

        return pd.DataFrame(records, columns=ref_names + ["ssr"])

    def choose_best_component_count(self, all_counts_spectrum_fit_table):
        """
        Calculate bootstrap validation statistics for the top 20 fits per component count,
        then select the component count whose best fit has the lowest 95% CI of median SSR.
        When confidence intervals overlap, prefer the lower component count.

        Parameters
        ----------
        all_counts_spectrum_fit_table : dict
            Keys are component counts, values are lists of SpectrumFit sorted by NSS.

        Returns
        -------
        SpectrumFit
        """
        component_counts = list(all_counts_spectrum_fit_table)
        a_fit = all_counts_spectrum_fit_table[component_counts[0]][0]
        log = logging.getLogger(__name__ + ":" + a_fit.unknown_spectrum.file_name)

        component_count_to_median_ssr = {
            cc: np.inf for cc in all_counts_spectrum_fit_table.keys()
        }
        component_count_to_median_ssr_ci_lo_hi = {
            cc: (np.inf, np.inf) for cc in all_counts_spectrum_fit_table.keys()
        }

        all_counts_spectrum_fit_bv_table = defaultdict(list)
        for component_count_i in sorted(all_counts_spectrum_fit_table.keys()):
            log.debug(
                "calculating bootstrap validation SSR CI for %d component(s)",
                component_count_i,
            )

            sorted_fits = sorted(
                all_counts_spectrum_fit_table[component_count_i],
                key=lambda fit: fit.nss,
            )

            for fit_j in sorted_fits[:20]:
                bootstrap_df = self.calculate_bootstrap_results(fit_j)
                fit_j.bootstrap_df = bootstrap_df

                ssr_values = bootstrap_df["ssr"].values
                fit_j.median_ssr = np.median(ssr_values)
                ci_lo, ci_hi = scikits.bootstrap.ci(
                    data=ssr_values, statfunction=np.median
                )
                fit_j.median_ssr_ci_lo = float(ci_lo)
                fit_j.median_ssr_ci_hi = float(ci_hi)

                coef_cols = [c for c in bootstrap_df.columns if c != "ssr"]
                coef_records = {}
                for col in coef_cols:
                    col_values = bootstrap_df[col].values
                    col_ci_lo, col_ci_hi = scikits.bootstrap.ci(
                        data=col_values, statfunction=np.median
                    )
                    coef_records[col] = {
                        "median": float(np.median(col_values)),
                        "ci_lo": float(col_ci_lo),
                        "ci_hi": float(col_ci_hi),
                    }
                fit_j.bootstrap_coef_ci_df = pd.DataFrame(coef_records).T

                all_counts_spectrum_fit_bv_table[component_count_i].append(fit_j)

            all_counts_spectrum_fit_bv_table[component_count_i] = sorted(
                all_counts_spectrum_fit_bv_table[component_count_i],
                key=lambda fit: (
                    fit.median_ssr,
                    fit.median_ssr_ci_lo,
                    fit.median_ssr_ci_hi,
                ),
            )

            best_for_count = all_counts_spectrum_fit_bv_table[component_count_i][0]
            component_count_to_median_ssr[component_count_i] = best_for_count.median_ssr
            component_count_to_median_ssr_ci_lo_hi[component_count_i] = (
                best_for_count.median_ssr_ci_lo,
                best_for_count.median_ssr_ci_hi,
            )

            log.debug(
                "component count %d: best SSR CI %8.3f <-- %8.3f --> %8.3f",
                component_count_i,
                best_for_count.median_ssr_ci_lo,
                best_for_count.median_ssr,
                best_for_count.median_ssr_ci_hi,
            )

        best_component_count, _, _ = PredictionErrorFitTask.get_best_ci_component_count(
            component_count_to_median_ssr, component_count_to_median_ssr_ci_lo_hi
        )
        best_fit = all_counts_spectrum_fit_table[best_component_count][0]
        log.info("best fit: {}".format(best_fit))
        return best_fit

    def plot_top_fits(self, spectrum, fit_results):
        figure_list = []

        top_fit_per_component_count = {}
        for component_count in fit_results.component_count_fit_table.keys():
            bv_fits = [
                fit
                for fit in fit_results.component_count_fit_table[component_count]
                if hasattr(fit, "median_ssr")
            ]
            if not bv_fits:
                continue

            sorted_fits = sorted(bv_fits, key=lambda fit: fit.median_ssr)[:10]
            top_fit_per_component_count[component_count] = sorted_fits[0]

            f, ax = plt.subplots()
            bootstrap_validation_box_plots(
                ax=ax,
                title=f"Best {component_count}-component Fits\n{spectrum.file_name}",
                sorted_fits=sorted_fits,
            )
            f.tight_layout()
            figure_list.append(f)

        if top_fit_per_component_count:
            f, ax = plt.subplots()
            best_bootstrap_fit_for_component_count_box_plots(
                ax=ax,
                title=f"Best Fits\n{spectrum.file_name}",
                top_fit_per_component_count=top_fit_per_component_count,
            )
            f.tight_layout()
            figure_list.append(f)

        return figure_list
