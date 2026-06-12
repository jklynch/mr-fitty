from collections import defaultdict
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from mrfitty.base import AdaptiveEnergyRangeBuilder
from mrfitty.combination_fit import AllCombinationFitTask
from mrfitty.linear_model import OlsWithStats
from mrfitty.plot import (
    bootstrap_validation_box_plots,
    best_bootstrap_fit_for_component_count_box_plots,
)
from mrfitty.prediction_error_fit import PredictionErrorFitTask


class BootstrapValidationFitTask(AllCombinationFitTask):
    def __init__(
        self,
        reference_spectrum_list,
        unknown_spectrum_list,
        ls=OlsWithStats,
        energy_range_builder=AdaptiveEnergyRangeBuilder(),
        component_count_range=range(1, 4),
        best_fits_plot_limit=3,
        bootstrap_count=9999,
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
                any_given_fit.ssr_ci_lo,
                any_given_fit.median_ssr,
                any_given_fit.ssr_ci_hi,
            ),
            "MSE: {:<8.3f}".format(any_given_fit.nss),
        ]

    def calculate_bootstrap_statistics(self, fit):
        """
        Split the spectrum into even-indexed (training) and odd-indexed (validation)
        data sets. Run self.bootstrap_count bootstrap fits using the method of resampled
        residuals. For each fit record the reference coefficients and the sum of squared
        residuals on the validation set.

        Calculate 95% confidence intervals of the median of the bootstrapped
        coefficients and normalized sum of squared residuals. In particular the
        normalized ssr distibution is one-tailed and standard methods of determining
        the confidence interval of the mean do not work well.

        The return value is a pandas.DataFrame such as this
                As2O3_ref_avg_als_cal.e	     ssr
            0                  0.997711 2.308403
            1                  1.000515 2.309184
            2                  0.987700 2.308011
            3                  0.933714 2.369647
            ...
        1000 rows × 2 columns

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
        ref_names_and_ssr = ref_names + ["ssr"]

        # training_model = self.ls()
        # training_model.fit(A[train_idx], b[train_idx])
        # training_model_b = training_model.predict(A[train_idx])
        # training_model_residuals = b[train_idx] - training_model_b
        training_coef, _, _, _ = np.linalg.lstsq(A[train_idx], b[train_idx])
        training_residuals = b[train_idx] - A[train_idx] @ training_coef

        rng = np.random.default_rng()
        bootstrap_res = rng.choice(
            training_residuals, size=(len(training_residuals), 9999), replace=True
        )
        # bootstrap_res has shape (N, 9999)
        # bootstrap_res.shape

        # coef has shape (len(ref_names), 9999)
        bootstrap_coefs, _, _, _ = np.linalg.lstsq(
            A[train_idx], np.atleast_2d(b[train_idx]).T + bootstrap_res
        )

        bootstrap_validation_ssr = np.sqrt(
            np.sum(
                np.square(
                    A[valid_idx] * bootstrap_coefs - np.atleast_2d(b[valid_idx]).T
                ),
                axis=0,
            )
        )

        bootstrap_distribution = np.hstack(
            (bootstrap_coefs.T, np.atleast_2d(bootstrap_validation_ssr).T)
        )
        bootstrap_distribution_df = pd.DataFrame(
            bootstrap_distribution, columns=ref_names_and_ssr
        )
        # bootstrap_distribution_df looks like
        #        As2O3_ref_avg_als_cal.e       ssr
        #  0                    0.933248  2.370635
        #  1                    1.073729  2.430369
        #  ...                       ...       ...
        #  9997                 0.971169  2.315552
        #  9998                 1.024857  2.328230
        #  [9999 rows x 2 columns])

        confidence = 0.95
        alpha = (1.0 - confidence) / 2.0
        alpha_interval = np.asarray(((alpha,), (1 - alpha,)))
        # alpha_interval looks like this
        #   [[0.025],
        #    [0.975]]

        bootstrap_percentile_ci_df = pd.DataFrame(
            data=np.vstack(
                (
                    scipy.stats.quantile(bootstrap_distribution, alpha_interval),
                    np.std(bootstrap_distribution, correction=1, axis=0, keepdims=True),
                )
            ),
            index=("ci_lower", "ci_upper", "stderr"),
            columns=ref_names_and_ssr,
        )
        # bootstrap_percentile_ci_df looks like
        #           As2O3_ref_avg_als_cal.e       ssr
        # ci_lower                 0.938335  2.307737
        # ci_upper                 1.055698  2.392696
        # stderr                   0.029756  0.024232

        return bootstrap_percentile_ci_df, bootstrap_distribution_df

    def calculate_bootstrap_statistics_classic(self, fit):
        """
        Split the spectrum into even-indexed (training) and odd-indexed (validation) data points.
        Run bootstrap_count bootstrapped fits on samples drawn with replacement from the
        training set. For each fit record the reference coefficients and the SSR on the
        validation set.

        The return value is a pandas.DataFrame such as this
                As2O3_ref_avg_als_cal.e	     ssr
            0                  0.997711 2.308403
            1                  1.000515 2.309184
            2                  0.987700 2.308011
            3                  0.933714 2.369647
            ...
        1000 rows × 2 columns

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
        ref_names_and_ssr = ref_names + ["ssr"]

        bootstrap_distribution = np.zeros(
            (self.bootstrap_count, len(ref_names_and_ssr)), dtype=np.float64
        )

        for i in range(self.bootstrap_count):
            boot_idx = np.random.choice(train_idx, size=len(train_idx), replace=True)
            lm = self.ls()
            lm.fit(A[boot_idx], b[boot_idx])
            ssr = np.sqrt(np.sum(np.square(lm.predict(A[valid_idx]) - b[valid_idx])))
            bootstrap_distribution[i, :-1] = lm.coef_
            bootstrap_distribution[i, -1] = ssr

        confidence = 0.95
        alpha = (1.0 - confidence) / 2.0
        alpha_interval = np.asarray(((alpha,), (1 - alpha,)))
        # alpha_interval looks like this
        #   [[0.025],
        #    [0.975]]

        bootstrap_percentile_ci_df = pd.DataFrame(
            data=np.vstack(
                (
                    scipy.stats.quantile(bootstrap_distribution, alpha_interval),
                    np.std(bootstrap_distribution, correction=1, axis=0, keepdims=True),
                )
            ),
            index=("ci_lower", "ci_upper", "stderr"),
            columns=ref_names_and_ssr,
        )
        # bootstrap_percentile_ci_df looks like
        #           As2O3_ref_avg_als_cal.e       ssr
        # ci_lower                 0.938335  2.307737
        # ci_upper                 1.055698  2.392696
        # stderr                   0.029756  0.024232

        bootstrap_distribution_df = pd.DataFrame(
            bootstrap_distribution, columns=ref_names_and_ssr
        )
        # bootstrap_distribution_df looks like
        #        As2O3_ref_avg_als_cal.e       ssr
        #  0                    0.933248  2.370635
        #  1                    1.073729  2.430369
        #  ...                       ...       ...
        #  9997                 0.971169  2.315552
        #  9998                 1.024857  2.328230
        #  [9999 rows x 2 columns])

        return bootstrap_percentile_ci_df, bootstrap_distribution_df

    def calculate_bootstrap_statistics_1(self, fit):
        """
        Split the spectrum into even-indexed (training) and odd-indexed (validation) data points.
        Run bootstrap_count bootstrapped fits on samples drawn with replacement from the
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

        def statistics(*idx):
            lm = self.ls()
            lm.fit(A[idx], b[idx])
            ssr = np.sqrt(np.sum(np.square(lm.predict(A[valid_idx]) - b[valid_idx])))
            return list(lm.coef_) + [ssr]

        bootstrap_results = scipy.stats.bootstrap(
            data=(train_idx,),
            statistic=statistics,
            n_resamples=self.bootstrap_count,
            method="percentile",
            confidence_level=0.95,
        )

        column_names = (*fit.reference_spectra_A_df.columns, "ssr")
        ci_df = pd.DataFrame(
            data=bootstrap_results.confidence_interval,
            index=["lower", "upper"],
            columns=column_names,
        )

        standard_error_df = pd.DataFrame(
            data=bootstrap_results.standard_error,
            index=column_names,
            columns=("standard error",),
        ).transpose()

        bootstrap_distribution_df = pd.DataFrame(
            data=bootstrap_results.bootstrap_distribution.T, columns=column_names
        )

        return ci_df, standard_error_df, bootstrap_distribution_df

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
        component_count_to_ssr_ci_lo_hi = {
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
                bootstrap_ci_df, bootstrap_df = self.calculate_bootstrap_statistics(
                    fit_j
                )

                fit_j.bootstrap_df = bootstrap_df

                fit_j.median_ssr = bootstrap_df["ssr"].median()
                fit_j.ssr_ci_lo = bootstrap_ci_df.loc["ci_lower", "ssr"]
                fit_j.ssr_ci_hi = bootstrap_ci_df.loc["ci_upper", "ssr"]

                reference_coef_records = {}
                for reference_coef_col in fit_j.reference_spectra_A_df.columns:
                    reference_coef_records[reference_coef_col] = {
                        "median": float(bootstrap_df[reference_coef_col].median()),
                        "ci_lo": float(
                            bootstrap_ci_df.loc["ci_lower", reference_coef_col]
                        ),
                        "ci_hi": float(
                            bootstrap_ci_df.loc["ci_upper", reference_coef_col]
                        ),
                    }
                fit_j.bootstrap_coef_ci_df = pd.DataFrame(reference_coef_records).T

                all_counts_spectrum_fit_bv_table[component_count_i].append(fit_j)

            all_counts_spectrum_fit_bv_table[component_count_i] = sorted(
                all_counts_spectrum_fit_bv_table[component_count_i],
                key=lambda fit: (
                    fit.median_ssr,
                    fit.ssr_ci_lo,
                    fit.ssr_ci_hi,
                ),
            )

            best_for_count = all_counts_spectrum_fit_bv_table[component_count_i][0]
            component_count_to_median_ssr[component_count_i] = best_for_count.median_ssr
            component_count_to_ssr_ci_lo_hi[component_count_i] = (
                best_for_count.ssr_ci_lo,
                best_for_count.ssr_ci_hi,
            )

            log.debug(
                "component count %d: best SSR CI %8.3f <-- %8.3f --> %8.3f",
                component_count_i,
                best_for_count.ssr_ci_lo,
                best_for_count.median_ssr,
                best_for_count.ssr_ci_hi,
            )

        best_component_count, _, _ = PredictionErrorFitTask.get_best_ci_component_count(
            component_count_to_median_ssr, component_count_to_ssr_ci_lo_hi
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
