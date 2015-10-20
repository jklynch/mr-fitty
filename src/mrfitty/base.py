"""

"""
import logging
import os.path
import re

import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

log = logging.getLogger(name=__name__)


class Spectrum:
    def __init__(self, file_path, data_df):
        self.file_path = file_path
        self.file_name = os.path.split(file_path)[1]
        self.data_df = data_df

    def __repr__(self):
        return 'Spectrum({}, {})'.format(self.file_path, self.data_df.shape)

    @classmethod
    def read_file(cls, file_path_or_buffer, **kwargs):
        spectrum_data_df = pd.read_csv(
            file_path_or_buffer,
            engine='python',
            #sep='\t',
            sep='[ \t]+',
            comment='#',
            header=None
        )
        log.info('read {}'.format(file_path_or_buffer))
        log.debug('  shape is {}'.format(spectrum_data_df.shape))
        if spectrum_data_df.shape[1] < 2:
            raise Exception('{} has fewer than 2 columns'.format(file_path_or_buffer))
        if np.isnan(spectrum_data_df.values).any():
            raise Exception('{} has one or more NaN values'.format(file_path_or_buffer))
        spectrum_data_df = spectrum_data_df.iloc[:, :2]
        # try to assign names to the first two columns
        spectrum_data_df.columns = ['energy', 'norm']
        log.debug('  first incident energy is {}'.format(spectrum_data_df.energy.iloc[0]))
        log.debug('  last incident energy is  {}'.format(spectrum_data_df.energy.iloc[-1]))
        return cls(file_path_or_buffer, spectrum_data_df, **kwargs)


class ReferenceSpectrum(Spectrum):
    def __init__(self, file_path, reference_spectrum_data, mineral_category=None):
        super(type(self), self).__init__(file_path, reference_spectrum_data)
        self.mineral_category = mineral_category
        self.interpolant = InterpolatedUnivariateSpline(
            reference_spectrum_data.energy.values,
            reference_spectrum_data.norm.values
        )

    def __repr__(self):
        return 'ReferenceSpectrum({}, {}, {})'.format(self.file_path, self.data_df.shape, self.mineral_category)


class SpectrumFit:
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
        self.unknown_spectrum_b = unknown_spectrum_b
        self.reference_spectra_coef_x = reference_spectra_coef_x
        self.fit_spectrum_b = reference_spectra_A_df.dot(reference_spectra_coef_x)
        self.residuals = self.fit_spectrum_b - self.unknown_spectrum_b
        self.sum_of_abs_residuals = np.sum(np.abs(self.residuals))
        self.sum_of_abs_unknown_spectum_b = np.sum(np.abs(unknown_spectrum_b))
        self.sum_of_squared_residuals = np.sum(np.power(self.residuals, 2.0))
        self.sum_of_squared_unknown_spectrum_b = np.sum(np.power(unknown_spectrum_b, 2.0))
        self.nsa = self.sum_of_abs_residuals / self.sum_of_abs_unknown_spectum_b
        self.nss = self.sum_of_squared_residuals / self.sum_of_squared_unknown_spectrum_b

        # calculate the approximate area under each curve
        self.residuals_auc = np.sum(np.abs(self.residuals))
        self.unknown_spectrum_auc = np.sum(np.abs(self.unknown_spectrum_b))

        # calculate percent contribution for each reference
        #
        #  pct_i = coef_i * auc_ref_i * (100.0 / unknown_spectrum_auc)
        #
        self.reference_contribution_percent_sr = \
            (100.0 / self.unknown_spectrum_auc) \
            * self.reference_spectra_coef_x \
            * np.sum(np.abs(self.reference_spectra_A_df), axis=0)

        log.debug('reference_contribution_percent:\n{}'.format(self.reference_contribution_percent_sr))
        self.total_reference_contribution = np.sum(self.reference_contribution_percent_sr)
        log.debug('total_reference_contribution: {}'.format(self.total_reference_contribution))
        self.residuals_contribution = (100.0 / self.unknown_spectrum_auc) * self.residuals_auc
        log.debug('residuals_contribution: {}'.format(self.residuals_contribution))

    def __str__(self):
        contribution_str = 'SpectrumFit: {:5.3f}: NSS'.format(self.nss)
        for reference_spectrum in self.reference_spectra_seq:
            # TODO: something smarter than reference_spectrum.file_name
            reference_contribution = self.reference_contribution_percent_sr[reference_spectrum.file_name]
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
