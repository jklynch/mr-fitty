from glob import glob
import os.path

import numpy as np
import pandas as pd
import pytest

import mrfitty
from mrfitty.base import ReferenceSpectrum, Spectrum


@pytest.fixture(scope="module")
def synthetic_spectra():
    energies = np.linspace(11800, 11900, 50)
    ref_a_norm = np.sin(energies / 100)
    ref_b_norm = np.cos(energies / 100)

    ref_a_df = pd.DataFrame(
        {"norm": ref_a_norm}, index=pd.Index(energies, name="energy")
    )
    ref_b_df = pd.DataFrame(
        {"norm": ref_b_norm}, index=pd.Index(energies, name="energy")
    )
    ref_a = ReferenceSpectrum("ref_a.e", ref_a_df, {})
    ref_b = ReferenceSpectrum("ref_b.e", ref_b_df, {})

    coef = np.array([0.6, 0.4])
    unknown_norm = (
        coef[0] * ref_a_norm
        + coef[1] * ref_b_norm
        + np.random.default_rng(0).normal(0, 0.001, 50)
    )
    unknown_df = pd.DataFrame(
        {"norm": unknown_norm}, index=pd.Index(energies, name="energy")
    )
    unknown = Spectrum("unknown.e", unknown_df, {})

    return [ref_a, ref_b], unknown


@pytest.fixture(scope="module")
def arsenic_example_path():
    mrfitty_init_fp = mrfitty.__file__
    mrfitty_package_dir_path, _ = os.path.split(mrfitty_init_fp)
    mrfitty_dir_path, _ = os.path.split(mrfitty_package_dir_path)

    return os.path.join(mrfitty_dir_path, "example", "arsenic")


@pytest.fixture(scope="module")
def arsenic_references(arsenic_example_path):
    reference_file_path_pattern = os.path.join(arsenic_example_path, "reference", "*.e")

    reference_spectrum_list = [
        ReferenceSpectrum.read_file(file_path)
        for file_path in glob(reference_file_path_pattern)
    ]

    return reference_spectrum_list


@pytest.fixture(scope="module")
def arsenic_unknowns(arsenic_example_path):
    data_file_path_pattern = os.path.join(arsenic_example_path, "unknown", "*.e")

    unknown_spectrum_list = [
        Spectrum.read_file(file_path) for file_path in glob(data_file_path_pattern)
    ]

    return unknown_spectrum_list
