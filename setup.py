#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='mrfitty',
    version='0.13.1',
    license='MIT',
    description='A package for linear least squares fitting XANES data.',
    long_description='A package for linear least squares fitting XANES data.',
    author='Joshua Lynch',
    author_email='joshua.kevin.lynch@gmail.com',
    url='https://github.com/jklynch/mr-fitty',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Utilities',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        'bokeh',
        'jupyter',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'scikits.bootstrap',
        'scikit-learn'
    ],
    data_files=[
        ('sample_data/reference', glob('experiment_arsenic/reference_spectra/*.e')),
        ('sample_data/unknown', glob('experiment_arsenic/unknown_spectra/*.e'))
    ],
    extras_require={
        'test': ['pytest', 'pyfakefs', 'coverage', 'pytest-catchlog', 'pytest-cov'],
    },
    entry_points={
        'console_scripts': [
            'mrfitty = mrfitty.__main__:main',
        ]
    },
)
