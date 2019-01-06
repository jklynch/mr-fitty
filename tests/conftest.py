"""
The MIT License (MIT)

Copyright (c) 2015-2018 Joshua Lynch, Sarah Nicholas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from glob import glob
import os.path

import pytest

import mrfitty
from mrfitty.base import ReferenceSpectrum, Spectrum
from mrfitty.database import FitDatabase


@pytest.fixture(scope='module')
def arsenic_example_path():
    mrfitty_init_fp = mrfitty.__file__
    mrfitty_dir_path, _ = os.path.split(mrfitty_init_fp)
    src_dir_path, _ = os.path.split(mrfitty_dir_path)

    return os.path.join(src_dir_path, 'example', 'arsenic')


@pytest.fixture(scope="module")
def arsenic_references(arsenic_example_path):
    reference_file_path_pattern = os.path.join(arsenic_example_path, 'reference', '*.e')

    reference_spectrum_list = [
        ReferenceSpectrum.read_file(file_path)
        for file_path
        in glob(reference_file_path_pattern)
    ]

    return reference_spectrum_list


@pytest.fixture(scope='module')
def arsenic_unknowns(arsenic_example_path):
    mrfitty_init_fp = mrfitty.__file__
    mrfitty_dir_path, _ = os.path.split(mrfitty_init_fp)
    src_dir_path, _ =os.path.split(mrfitty_dir_path)

    data_file_path_pattern = os.path.join(arsenic_example_path, 'unknown', '*.e')

    unknown_spectrum_list = [
        Spectrum.read_file(file_path)
        for file_path
        in glob(data_file_path_pattern)
    ]

    return unknown_spectrum_list


@pytest.fixture
def fit_db():
    fit_db = FitDatabase(url='sqlite:///:memory:', echo=True)
    fit_db.create_tables()
    return fit_db
