"""
The MIT License (MIT)

Copyright (c) 2015 Joshua Lynch, Sarah Nicholas

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
import glob
import logging
import os.path
import re

import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

log = logging.getLogger(name=__name__)


class Spectrum:
    """Spectrum

    Encapsulates spectrum data contained in TODO: files.

    Parameters
    ----------
    file_path : str, required
        Path to the file containing this spectrum.

    data_df : pandas.Dataframe, required
        Array of incident energies and fluorescence TODO: values??
        (index)    incident_energy    fluorescence
           0         1000.1             0.001
           1         1000.2             0.002
          ...        ...                ...
          100        1100.3             0.100

    Attributes
    ----------
    file_path : str

    file_name : str

    data_df : pandas.Dataframe
    """
    def __init__(self, file_path, data_df):
        self.file_path = file_path
        self.file_name = os.path.split(file_path)[1]
        self.data_df = data_df

    def __repr__(self):
        return 'Spectrum({}, {})'.format(self.file_path, self.data_df.shape)

    @classmethod
    def read_file(cls, file_path_or_buffer, **kwargs):
        """Read a file called what?
        TODO: what are these files called?
        Parameters
        ----------
        cls : Spectrum or Spectrum subclass, required
            The class to be instantiated with contents of the specified file.

        file_path_or_buffer : path to file or file-like object, required
            The file to be read.

        **kwargs : keyword arguments, optional
            Arguments for the cls constructor.

        Returns
        -------
        Instance of class Spectrum or subclass.
        """
        spectrum_data_df = pd.read_csv(
            file_path_or_buffer,
            engine='python',
            sep='[ \t]+',
            comment='#',
            header=None
        )
        log.info('read {}'.format(file_path_or_buffer))
        log.debug('  shape is {}'.format(spectrum_data_df.shape))
        if spectrum_data_df.shape[1] < 2:
            raise Exception('{} has fewer than 2 columns'.format(file_path_or_buffer))
        # keep only the first two columns
        spectrum_data_df = spectrum_data_df.iloc[:, :2]
        if np.isnan(spectrum_data_df.values).any():
            raise Exception('{} has one or more NaN values'.format(file_path_or_buffer))
        # try to assign names to the first two columns
        spectrum_data_df.columns = ['energy', 'norm']
        spectrum_data_df.index = spectrum_data_df.energy
        log.debug('  first incident energy is {}'.format(spectrum_data_df.energy.iloc[0]))
        log.debug('  last incident energy is  {}'.format(spectrum_data_df.energy.iloc[-1]))
        return cls(file_path_or_buffer, spectrum_data_df, **kwargs)

    @classmethod
    def read_all(cls, file_glob_list):
        """
        Return a ReferenceSpectrum instance for each file in the file glob list.
        :param file_glob_list: list of paths or file globs
        :return: set of ReferenceSpectrum instances
        """
        # keep a list of config file entries for error reporting
        reference_spectrum_file_path_set = set()

        # return this set of duplicate file paths
        duplicate_file_path_set = set()
        # return this set of ReferenceSpectrum instances
        reference_spectrum_set = set()
        for reference_spectrum_file_glob in file_glob_list:
            logging.info('reference file pattern: {}'.format(reference_spectrum_file_glob))
            reference_spectrum_file_glob_expanded = os.path.expanduser(reference_spectrum_file_glob)
            #reference_spectrum_file_glob_list.append(reference_spectrum_file_glob_expanded)
            logging.info('expanded file pattern: {}'.format(reference_spectrum_file_glob_expanded))
            for i, reference_spectrum_file_path in enumerate(glob.glob(reference_spectrum_file_glob_expanded)):
                if reference_spectrum_file_path in reference_spectrum_file_path_set:
                    logging.info('  reference file {} has already been read'.format(reference_spectrum_file_path))
                    duplicate_file_path_set.add(reference_spectrum_file_path)
                else:
                    log.info('  reading reference file {}: {}'.format(i, reference_spectrum_file_path))
                    reference_spectrum = cls.read_file(reference_spectrum_file_path)
                    reference_spectrum_set.add(reference_spectrum)

        return reference_spectrum_set, duplicate_file_path_set


class ReferenceSpectrum(Spectrum):
    """ReferenceSpectrum

    Parameters
    ----------
    file_path : str

    reference_spectrum_data : pandas.Dataframe

    mineral_category : str

    Attributes
    ----------
    mineral_category : str

    interpolant : scipy.interpolate.InterpolatedUnivariateSpline
    """
    def __init__(self, file_path, reference_spectrum_data, mineral_category=None):
        super(type(self), self).__init__(file_path, reference_spectrum_data)
        self.mineral_category = mineral_category
        self.interpolant = InterpolatedUnivariateSpline(
            reference_spectrum_data.energy.values,
            reference_spectrum_data.norm.values
        )

    def __repr__(self):
        return 'ReferenceSpectrum({}, {}, {})'.format(self.file_path, self.data_df.shape, self.mineral_category)


class InterpolatedReferenceSpectraSet:
    def __init__(self, unknown_spectrum, reference_set):
        self.unknown_spectrum = unknown_spectrum
        self.interpolated_reference_set_df = InterpolatedReferenceSpectraSet.get_interpolated_reference_set_df(
            unknown_spectrum=unknown_spectrum,
            reference_set=reference_set)

    #@profile
    def get_reference_subset_and_unknown_df(self, reference_list):
        reference_name_list = sorted([r.file_name for r in reference_list])
        keep_rows = self.interpolated_reference_set_df.loc[:, reference_name_list].notnull().all(axis=1)
        reference_subset_df = self.interpolated_reference_set_df.loc[keep_rows.values, reference_name_list]
        unknown_subset_df = self.unknown_spectrum.data_df.loc[reference_subset_df.index]
        return reference_subset_df, unknown_subset_df

    @staticmethod
    def get_interpolated_reference_set_df(unknown_spectrum, reference_set):
        # the interpolated reference spectra will be unknown_spectrum.data_df.shape[0] x len(reference_set)
        interpolated_reference_spectra = np.zeros((unknown_spectrum.data_df.shape[0], len(reference_set)))
        column_names = []
        for i, reference_spectrum in enumerate(sorted(list(reference_set), key=lambda r: r.file_name)):
            column_names.append(reference_spectrum.file_name)
            interpolated_reference_spectra[:, i] = reference_spectrum.interpolant(
                unknown_spectrum.data_df.energy)
            ndx = InterpolatedReferenceSpectraSet.get_extrapolated_value_index(
                unknown_energy=unknown_spectrum.data_df.energy.values,
                reference_energy=reference_spectrum.data_df.energy.values)
            # print(ndx)
            interpolated_reference_spectra[ndx, i] = np.nan

        interpolated_reference_spectra_df = pd.DataFrame(
            data=interpolated_reference_spectra,
            index=unknown_spectrum.data_df.energy,
            columns=column_names)

        return interpolated_reference_spectra_df

    @staticmethod
    def get_extrapolated_value_index(unknown_energy, reference_energy):
        extrapolated_value_boolean_index = np.logical_or(
            unknown_energy < reference_energy[0],
            unknown_energy > reference_energy[-1] )
        return np.where(extrapolated_value_boolean_index)


class SpectrumFit:
    """SpectrumFit

    Encapsulates a single fit of one or more reference spectra to a single unknown spectrum.

    Parameters
    ----------
    interpolant_incident_energy :

    reference_spectra_A_df :

    unknown_spectrum_b :

    reference_spectra_seq :

    reference_spectra_coef_x :

    Attributes
    ----------
    interpolant_incident_energy :

    reference_spectra_A_df :

    unknown_spectrum_b :

    reference_spectra_seq :

    reference_spectra_coef_x :

    """
    #@profile
    def __init__(
        self,
        interpolant_incident_energy,
        reference_spectra_A_df,
        unknown_spectrum_b,
        reference_spectra_seq,
        reference_spectra_coef_x
    ):
        self.interpolant_incident_energy = interpolant_incident_energy
        self.reference_spectra_seq = reference_spectra_seq
        self.reference_spectra_A_df = reference_spectra_A_df
        # TODO: fix this
        self.unknown_spectrum_b = unknown_spectrum_b.norm
        self.reference_spectra_coef_x = reference_spectra_coef_x
        self.fit_spectrum_b = reference_spectra_A_df.dot(reference_spectra_coef_x)
        self.residuals = self.fit_spectrum_b - self.unknown_spectrum_b
        log.debug('self.unknown_spectrum_b :\n%s', self.unknown_spectrum_b)
        log.debug('self.fit_spectrum_b     :\n%s', self.fit_spectrum_b)
        log.debug('residuals               :\n%s', self.residuals)

        self.sum_of_abs_residuals = np.sum(np.abs(self.residuals))
        self.sum_of_abs_unknown_spectrum_b = np.sum(np.abs(self.unknown_spectrum_b))
        self.sum_of_squared_residuals = np.sum(np.power(self.residuals, 2.0))
        self.sum_of_squared_unknown_spectrum_b = np.sum(np.power(self.unknown_spectrum_b, 2.0))

        self.nsa = self.sum_of_abs_residuals / self.sum_of_abs_unknown_spectrum_b
        self.nss = self.sum_of_squared_residuals / self.sum_of_squared_unknown_spectrum_b

        self.reference_contribution_percent_sr = None
        self.total_reference_contribution = None
        self.residuals_contribution = None

        # calculate the approximate area under each curve
        self.residuals_auc = self.sum_of_abs_residuals
        self.unknown_spectrum_auc = self.sum_of_abs_unknown_spectrum_b

    def get_reference_contributions_sr(self):

        # calculate percent contribution for each reference
        #
        #  pct_i = coef_i * auc_ref_i * (100.0 / unknown_spectrum_auc)
        #
        self.reference_contribution_percent_sr = \
            (100.0 / self.unknown_spectrum_auc) \
            * self.reference_spectra_coef_x \
            * np.sum(np.abs(self.reference_spectra_A_df), axis=0)

        log.debug('reference_contribution_percent:\n%s', self.reference_contribution_percent_sr)
        self.total_reference_contribution = np.sum(self.reference_contribution_percent_sr)
        log.debug('total_reference_contribution: %s', self.total_reference_contribution)
        self.residuals_contribution = (100.0 / self.unknown_spectrum_auc) * self.residuals_auc
        log.debug('residuals_contribution: %s', self.residuals_contribution)
        return self.reference_contribution_percent_sr

    def __str__(self):
        contribution_str = 'SpectrumFit: {:5.3f}: NSS'.format(self.nss)
        reference_contribution_sr = self.get_reference_contributions_sr()
        for reference_spectrum in self.reference_spectra_seq:
            # TODO: something smarter than reference_spectrum.file_name
            reference_contribution = reference_contribution_sr[reference_spectrum.file_name]
            contribution_str += ' {:4.2f}: {},'.format(reference_contribution, reference_spectrum.file_name)
        contribution_str += ' {:4.2f}: residuals'.format(self.residuals_contribution)
        return contribution_str

"""
A .prm file looks like this:
    NbCompoMax=3
    NbCompoMin=1
    # means not taken into account
    ref=Aegirine.e
    ref=Akaganeite.e
    #ref=Al184_Fe016.e
    #ref=Al192_Fe008.e
    #ref=Al198_Fe002.e
    ref=Almandine.e
    ref=Andradite.e
    ref=Augite.e
    Ref = ""

What does that last line mean?
"""

nb_combo_max_pattern = re.compile(r'NbCompoMax=(?P<NbCompoMax>(\d+))', re.IGNORECASE)
nb_combo_min_pattern = re.compile(r'NbCompoMin=(?P<NbCompoMin>(\d+))', re.IGNORECASE)
reference_file_name_pattern = re.compile(r'ref=(?P<ref>\S+)(\s+(?P<mineral_category>\S+))?')


class PRM:
    def __init__(self, nb_compo_max, nb_compo_min):
        self.check_component_range(nb_compo_max, nb_compo_min)
        self.nb_component_max = nb_compo_max
        self.nb_component_min = nb_compo_min
        self.component_count_range = range(nb_compo_min, nb_compo_max + 1)
        self.reference_file_path_list = []
        self.reference_file_path_to_mineral_category = {}

    def add_reference_file_path(self, reference_file_path, mineral_category=None):
        if reference_file_path in self.reference_file_path_list:
            #raise Exception('{} appears more than once in the prm'.format(reference_file_path))
            log.warning('{} appears more than once in prm'.format(reference_file_path))
        else:
            self.reference_file_path_list.append(reference_file_path)
            self.reference_file_path_to_mineral_category[reference_file_path] = mineral_category

    def get_reference_count(self):
        return len(self.reference_file_path_list)

    def check_component_range(self, nb_compo_max, nb_compo_min):
        if 0 < nb_compo_min <= nb_compo_max:
            pass
        else:
            raise Exception('unreasonable component counts:\n  NbCompoMin={}\n  NbCompoMax={}'.format(
                nb_compo_max,
                nb_compo_min
            ))

    @classmethod
    def read_prm(cls, prm_file_path):
        reference_dir_path = os.path.split(prm_file_path)[0]
        log.debug('reading PRM {}'.format(prm_file_path))
        with open(prm_file_path) as prm_file:
            nb_compo_max_match = nb_combo_max_pattern.match(prm_file.readline())
            nb_compo_max = int(nb_compo_max_match.group('NbCompoMax'))
            nb_compo_min_match = nb_combo_min_pattern.match(prm_file.readline())
            nb_compo_min = int(nb_compo_min_match.group('NbCompoMin'))
            prm = PRM(nb_compo_max, nb_compo_min)
            for ref_line in prm_file.readlines():
                ref_line = ref_line.strip()
                print(ref_line)
                if len(ref_line) == 0:
                    pass
                elif ref_line.startswith('#'):
                    pass
                elif ref_line == 'Ref = ""':
                    pass
                else:
                    reference_file_name_pattern_match = reference_file_name_pattern.match(ref_line)
                    if reference_file_name_pattern_match is None:
                        raise Exception('failed to recognize {} in prm {}'.format(ref_line, prm_file_path))
                    else:
                        reference_file_name = reference_file_name_pattern_match.group('ref')
                        reference_file_path = os.path.join(reference_dir_path, reference_file_name)
                        mineral_category = reference_file_name_pattern_match.group('mineral_category')
                        if not os.path.isfile(reference_file_path):
                            error_msg = 'reference file {} does not exist'.format(reference_file_path)
                            log.error(error_msg)
                            raise Exception(error_msg)
                        else:
                            log.debug('reading reference path {}'.format(reference_file_path))
                            prm.add_reference_file_path(reference_file_path, mineral_category)

        # we would like to be certain that either:
        # all reference files have a mineral category
        # or
        # no reference file has a mineral category
        unique_mineral_categories = set(prm.reference_file_path_to_mineral_category.values())
        if len(unique_mineral_categories) > 1 and (None in unique_mineral_categories):
            log.error(prm.reference_file_path_to_mineral_category)
            log.error('some reference files do not have a mineral category')
            raise Exception('some reference files do not have a mineral category')
        else:
            pass
        return prm
