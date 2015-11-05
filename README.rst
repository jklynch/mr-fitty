=======
MrFitty
=======

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - |version| |downloads|

.. |docs| image:: https://readthedocs.org/projects/mr-fitty/badge/?style=flat
    :target: https://readthedocs.org/projects/mr-fitty
    :alt: Documentation Status

.. |travis| image:: https://img.shields.io/travis/jklynch/mr-fitty/master.svg?style=flat&label=Travis
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/jklynch/mr-fitty

.. |codecov| image:: https://img.shields.io/codecov/c/github/jklynch/mr-fitty/master.svg?style=flat&label=Codecov
    :alt: Coverage Status
    :target: https://codecov.io/github/jklynch/mr-fitty

.. |version| image:: https://img.shields.io/pypi/v/mrfitty.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/mrfitty

.. |downloads| image:: https://img.shields.io/pypi/dm/mrfitty.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/mrfitty

A package for linear least squares fitting XANES data.  Generated with https://github.com/ionelmc/cookiecutter-pylibrary

* Free software: MIT license

Requirements
============

MrFitty requires Python 3.4 or higher and the following packages:

    1. click >= 5.1
    2. matplotlib >= 1.4.3
    3. numpy >= 1.10
    4. pandas >= 0.16
    5. scipy >= 0.16

Installation
============

Users should consider installing MrFitty in a `Python virtual environment <https://docs.python.org/3.4/library/venv.html>`_.
This is not necessary but it simplifies package management.  On systems without Python 3.4 (or higher) the `Anaconda <https://anaconda.org>`_
Python distribution is a good choice that can be installed without administrative priviledges and also supports virtual environments.

Once Python is available MrFitty can be installed with pip: ::

    $ pip install mrfitty

Required packages will be automatically installed by pip.

Usage
=====

MrFitty runs from the command line.  Use the --help option to display usage: ::

    $ mrfitty --help
    Usage: mrfitty [OPTIONS] CONFIG_FP

    Options:
      --help   Show this message and exit.

The required CONFIG_FP argument specifies the path to a user-generated configuration file, e.g. ::

    $ mrfitty ~/mrfitty_arsenic_spectra.cfg

Here is an example configuration file: ::

    [references]
    prm =  reference/As_database_for_llsq_25_refs.prm

    [data]
    data/*.e

    [output]
    plots_pdf_fp = test_arsenic_fit_plots.pdf
    table_fp = test_arsenic_fit_table.txt



Documentation
=============

https://mr-fitty.readthedocs.org/

Development
===========

To run all tests::

    tox
