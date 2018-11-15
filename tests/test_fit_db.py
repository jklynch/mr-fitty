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
from tests import test_spectrum_fit


def test_spec_insert_query_fit(fit_db):
    """
    Insert fits with 1, 2, 3 components.

    """
    fit_1 = test_spectrum_fit.generate_spectrum_fit(reference_count=1)
    fit_2 = test_spectrum_fit.generate_spectrum_fit(reference_count=2)
    fit_3 = test_spectrum_fit.generate_spectrum_fit(reference_count=3)

    fit_db.insert(fit_1)


def test_spec_query_fit():
    """
    Query for
    """
    pass


def test_spec_file_hash():
    pass
