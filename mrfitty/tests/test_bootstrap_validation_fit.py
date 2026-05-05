import pytest
from mrfitty.bootstrap_validation_fit import BootstrapValidationFitTask
from mrfitty.combination_fit import CombinationFitResults


def _make_task(arsenic_references, arsenic_unknowns, bootstrap_count=10):
    return BootstrapValidationFitTask(
        reference_spectrum_list=arsenic_references[:3],
        unknown_spectrum_list=[arsenic_unknowns[0]],
        bootstrap_count=bootstrap_count,
    )


def test_fit_returns_best_fit(arsenic_references, arsenic_unknowns):
    task = _make_task(arsenic_references, arsenic_unknowns)
    best_fit, _ = task.fit(arsenic_unknowns[0])
    assert best_fit is not None


def test_best_fit_has_bootstrap_stats(arsenic_references, arsenic_unknowns):
    task = _make_task(arsenic_references, arsenic_unknowns)
    best_fit, _ = task.fit(arsenic_unknowns[0])

    assert hasattr(best_fit, "median_ssr")
    assert hasattr(best_fit, "median_ssr_ci_lo")
    assert hasattr(best_fit, "median_ssr_ci_hi")
    assert hasattr(best_fit, "bootstrap_df")
    assert hasattr(best_fit, "bootstrap_coef_ci_df")

    assert "ssr" in best_fit.bootstrap_df.columns
    assert len(best_fit.bootstrap_df) == 10  # bootstrap_count

    assert best_fit.median_ssr_ci_lo <= best_fit.median_ssr <= best_fit.median_ssr_ci_hi

    assert set(best_fit.bootstrap_coef_ci_df.columns) == {"median", "ci_lo", "ci_hi"}
    assert len(best_fit.bootstrap_coef_ci_df) == len(best_fit.reference_spectra_seq)


def test_plot_top_fits_returns_figures(arsenic_references, arsenic_unknowns):
    task = _make_task(arsenic_references, arsenic_unknowns)
    best_fit, fit_table = task.fit(arsenic_unknowns[0])
    fit_results = CombinationFitResults(
        spectrum=arsenic_unknowns[0],
        best_fit=best_fit,
        component_count_fit_table=fit_table,
    )
    figure_list = task.plot_top_fits(spectrum=arsenic_unknowns[0], fit_results=fit_results)
    assert len(figure_list) > 0


def test_get_fit_quality_score_text(arsenic_references, arsenic_unknowns):
    task = _make_task(arsenic_references, arsenic_unknowns)
    best_fit, _ = task.fit(arsenic_unknowns[0])
    text_lines = task.get_fit_quality_score_text(best_fit)
    assert len(text_lines) == 2
    assert "Bootstrap SSR" in text_lines[0]
    assert "MSE" in text_lines[1]
