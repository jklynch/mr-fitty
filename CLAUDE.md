# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install for development:**
```bash
pip install -e .
pip install -r requirements-dev.txt
```

**Run tests:**
```bash
pytest -s -v mrfitty/tests/
```

**Run a single test:**
```bash
pytest -s -v mrfitty/tests/test_foo.py::TestClass::test_method
```

**Run tests with coverage:**
```bash
pytest -s --cov=mrfitty mrfitty/tests/
```

**Lint:**
```bash
flake8 mrfitty/
```

**Format:**
```bash
black mrfitty/
```

Max line length is 115 (configured in `.flake8`).

## Architecture

MrFitty fits XANES (X-ray Absorption Near Edge Structure) spectra by finding the linear combination of reference spectra that best explains an unknown spectrum. The key differentiator is cross-validation-based prediction error for model selection, which avoids overfitting when selecting how many reference components to include.

### Data flow

```
INI config file
    → fit_task_builder.build_fit_task()
    → PredictionErrorFitTask or AllCombinationFitTask
    → read spectra (base.Spectrum / ReferenceSpectrum)
    → build energy range (AdaptiveEnergyRangeBuilder or FixedEnergyRangeBuilder)
    → interpolate to common grid (InterpolatedReferenceSpectraSet)
    → fit all combinations via NonNegativeLinearRegression (NNLS)
    → select best fit using prediction error / cross-validation
    → write PDF plots, TSV tables, per-spectrum fit files
```

### Key modules

- **`__main__.py`** — CLI entry point (`mrfitty` command). Parses args, calls `build_fit_task()`, runs the task, writes output.
- **`fit_task_builder.py`** — Parses the INI configuration file, builds the reference spectrum list, and instantiates the appropriate fit task object.
- **`base.py`** — Core data model: `Spectrum`, `ReferenceSpectrum`, `InterpolatedReferenceSpectraSet`, `SpectrumFit`, energy range builders (`AdaptiveEnergyRangeBuilder`, `FixedEnergyRangeBuilder`), and `PRM` (parameter/reference file parser).
- **`combination_fit.py`** — `AllCombinationFitTask`: exhaustively tests all combinations of reference spectra up to a configurable max component count.
- **`prediction_error_fit.py`** — `PredictionErrorFitTask`: wraps `AllCombinationFitTask` and adds bootstrap-based prediction error with confidence intervals to select the best model robustly.
- **`linear_model.py`** — `NonNegativeLinearRegression`: wraps `scipy.optimize.nnls` so fit coefficients are always ≥ 0.
- **`loss.py`** — `NormalizedSumOfSquares` and `PredictionError` (cross-validation loss used by `PredictionErrorFitTask`).
- **`plot.py`** — Generates matplotlib PDF output: fit overlays, residuals, reference spectra, prediction error statistics.
- **`database.py`** — Optional SQLAlchemy-based cache for fit results (`FitDatabase`).

### Configuration file format (INI)

Users run `mrfitty` with a config file that has sections `[fit]`, `[references]`, `[data]`, and `[output]`. `fit_task_builder.py` is the authoritative parser for this format.
