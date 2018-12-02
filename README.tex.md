# MrFitty

![TavisCI](https://travis-ci.org/jklynch/mr-fitty.svg?branch=develop)

MrFitty is an open-source Python package for fitting XANES data to a set of reference spectra using linear least
squares and best subset selection as described in *An Introduction to Statistical Learning with Applications in R* by
Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. It runs on any system with a Python 3.6+ interpreter
including Linux, OS X, and Windows.

MrFitty functions similarly to the spectra-fitting tools in the LabView software suite written by
Dr. Matthew Marcus at the Berkeley Synchrotron available [here](https://sites.google.com/a/lbl.gov/als-beamline1032/software-download>`).

* Free software: MIT license

## Overview

XANES spectrum fitting is a basic application of linear least squares: given the spectrum of an unknown sample and a library
of reference specta find the combination of references that best fit the unknown. Fitting each individual group of references
to the unknown is trivial, but selecting the 'best' combination of references is problematic because comparing fits with
differing numbers of reference spectra is not always straightforward.

A common measure of a fit's quality is the 'mean squared error' (MSE) defined by

$$MSE = \frac{1}{N} \sum_{i=1}^{N}(A_i-\hat{A}_i)^2$$

where the $A_i$ are the unknown spectrum and $\hat{A}_i$ are the fitted spectrum.

The problem in using MSE to compare, for example, the fit using references (X, Y)
to the fit using references (X, Y, Z) is that the 3-component fit will always have a better MSE than the 2-component fit.

## Requirements

MrFitty requires Python 3.6+. Required packages will be installed by pip.

## Installation

Users should consider installing MrFitty in a [Python virtual environment](https://docs.python.org/3.6/library/venv.html).
This is not necessary but it simplifies package management by separating the system Python from the users' Python environments.
On systems without Python 3.6+ the [Anaconda](https://anaconda.org) Python distribution is a good choice. It can be installed without
administrative privileges and supports virtual environments. In addition the Anaconda distribution includes pre-built
packages which are less trouble to install in some cases, especially on Windows systems.

### Method 1 (not recommended!)
If the standard Python 3.6+ distribution is available then MrFitty can be installed with pip:

    $ pip install git+https://github.com/jklynch/mr-fitty.git

### Method 2 (recommended)
Alternatively, MrFitty can be installed in a virtual environment using the standard Python 3.6+ distribution with the following commands:

    $ python3 -m venv mrf --without-pip
    $ source mrf/bin/activate
    (mrf) $ wget bootstrap.pypa.io/get-pip.py -O - | python3
    (mrf) $ pip install git+https://github.com/jklynch/mr-fitty.git

### Method 3 (recommended)
If the Anaconda distribution has been installed then MrFitty can be installed in a virtual environment with these commands:

    $ conda create python=3.6 --name mrf
    $ source activate mrf
    (mrf) $ conda install --file conda-requirements.txt
    (mrf) $ pip install git+https://github.com/jklynch/mr-fitty.git

In all cases the required packages will be automatically installed by pip.

## Update
Update MrFitty with pip as follows:

    (mrf) $ pip uninstall mrfitty
    (mrf) $ pip install git+https://github.com/jklynch/mr-fitty.git

## Usage
MrFitty runs from the command line.  The --help option will display usage instructions:

    $ mrfitty --help
    Usage: mrfitty [OPTIONS] CONFIG_FP

    Options:
      --help   Show this message and exit.

The required CONFIG_FP argument specifies the path to a configuration file written by the user, e.g.

    $ mrfitty ~/fit_arsenic_spectra.cfg

Here is an example configuration file that uses an existing PRM file:

    [fit]
    minimum_component_count = 1
    maximum_component_count = 3
    fit_method = lsq
    component_count_method = combination_fit

    [references]
    prm = reference/As_database_for_llsq_25_refs.prm

    [data]
    data/*.e

    [output]
    best_fit_files_dir = output
    plots_pdf_dir = output
    table_fp = output/fit_table.txt

    [plots]
    best_fits_plot_limit = 3

Here is an example configuration file that specifies reference files and fit parameters directly:

    [fit]
    minimum_component_count = 1
    maximum_component_count = 3
    fit_method = lsq
    component_count_method = combination_fit

    [references]
    reference/*.e

    [data]
    data/*.e

    [output]
    best_fit_files_dir = output
    plots_pdf_dir = output
    table_fp = output/fit_table.txt

    [plots]
    best_fits_plot_limit = 3

## Input
In addition to a configuration file the necessary input files are

  + at least two (but probably more) normalized reference spectrum files

  + one or more normalized spectrum files to be fit by the reference files

All input files must contain at least two columns of data. One column is
incident energy and the other column is normalized absorbance. Initial rows beginning
with '#' will be ignored except for the last row which must contain column headers.
For example, the following normalized file written by Athena can be used as
input to mr-fitty

    # Athena data file -- Athena version 0.8.056
    # Saving OTT3_55_spot0 as normalized mu(E)
    # .  Element=As   Edge=K
    # Background parameters
    # .  E0=11866.000  Eshift=0.000  Rbkg=1.000
    # .  Standard=0: None
    # .  Kweight=2.0  Edge step=0.004
    # .  Fixed step=no    Flatten=yes
    # .  Pre-edge range: [ -97.934 : -28.385 ]
    # .  Pre-edge line: 0.0068524 + -5.1098e-007 * E
    # .  Normalization range: [ 45.901 : 302.157 ]
    # .  Post-edge polynomial: 0.011539 + -5.5019e-007 * E + 0 * E^2
    # .  Spline range: [ 0.000 : 301.924 ]   Clamps: None/Strong
    # Foreward FT parameters
    # .  Kweight=0.5   Window=hanning   Phase correction=no
    # .  k-range: [ 2.000 : 6.902 ]   dk=1.00
    # Backward FT parameters
    # .  R-range: [ 1.000 : 3.000 ]
    # .  dR=0.00   Window=hanning
    # Plotting parameters
    # .  Multiplier=1   Y-offset=0.000
    # .
    #------------------------
    #  energy norm bkg_norm der_norm
      11766.697      -0.80039166E-02  -0.80037989E-02   0.61484012E-03
      11771.697      -0.49320333E-02  -0.49319157E-02   0.12085377E-02
      11776.697       0.40723503E-02   0.40724678E-02   0.17648519E-04
      ...

## Output
Several output files will be produced:

  + a single PDF containing plots of each fitted spectrum
  + a single table in text format with the best fit information for each fitted spectrum
  + one file per fitted spectrum with four columns of data:

     +  incident energy
     +  fitted normalized absorbance value
     +  residual of the fit
     +  input normalized absorbance
