from glob import glob
import os.path

import pytest

import mrfitty
from mrfitty.base import ReferenceSpectrum, Spectrum


@pytest.fixture(scope="module")
def arsenic_references():
    mrfitty_init_fp = mrfitty.__file__
    mrfitty_dir_path, _ = os.path.split(mrfitty_init_fp)
    src_dir_path, _ =os.path.split(mrfitty_dir_path)

    reference_file_path_pattern = os.path.join(src_dir_path, 'example', 'arsenic', 'reference', '*.e')

    reference_spectrum_list = [
        ReferenceSpectrum.read_file(file_path)
        for file_path
        in glob(reference_file_path_pattern)
    ]

    return reference_spectrum_list


@pytest.fixture(scope='module')
def arsenic_unknowns():
    mrfitty_init_fp = mrfitty.__file__
    mrfitty_dir_path, _ = os.path.split(mrfitty_init_fp)
    src_dir_path, _ =os.path.split(mrfitty_dir_path)

    data_file_path_pattern = os.path.join(src_dir_path, 'example', 'arsenic', 'unknown', '*.e')

    unknown_spectrum_list = [
        Spectrum.read_file(file_path)
        for file_path
        in glob(data_file_path_pattern)
    ]

    return unknown_spectrum_list
