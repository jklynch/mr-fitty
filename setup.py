#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import os
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
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
    version='0.1.0',
    license='MIT',
    description='A package for linear least squares fitting XANES data.',
    long_description='%s\n%s' % (read('README.rst'), re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))),
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Utilities',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        'bokeh',
        'click',
        'jupyter',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'scipy',
        'scikit-learn'
    ],
    data_files=[
        ('mrfitty_sample_ref', ['experiment_arsenic/reference_spectra/arsenate_aqueous_avg_als_cal.e']),
        ('mrfitty_sample_unknown', ['experiment_arsenic/unknown_spectra/OTT3_55_spot0.e'])
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'mrfitty = mrfitty.__main__:main',
        ]
    },
)
