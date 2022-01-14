from glob import glob
import os.path

import pytest

import mrfitty
from mrfitty.base import ReferenceSpectrum, Spectrum
from mrfitty.database import FitDatabase


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


@pytest.fixture
def fit_db():
    fit_db = FitDatabase(url="sqlite:///:memory:", echo=True)
    fit_db.create_tables()
    return fit_db
