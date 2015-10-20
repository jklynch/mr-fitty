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
import collections
import glob
import itertools
import logging
from operator import attrgetter
import os.path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.optimize import nnls

from mrfitty.base import PRM, ReferenceSpectrum, Spectrum, SpectrumFit


log = logging.getLogger(name=__name__)


class AdaptiveEnergyRangeBuilder:
    def __init__(self):
        pass

    def build_range(self, unknown_spectrum, reference_spectrum_seq):
        ref_min_last_energy = np.inf
        ref_max_first_energy = -1.0 * np.inf
        for s in reference_spectrum_seq:
            log.debug('s: {}'.format(s))
            if s.data_df.energy.iloc[-1] < ref_min_last_energy:
                ref_min_last_energy = s.data_df.energy.iloc[-1]
            else:
                pass
            if s.data_df.energy.iloc[0] > ref_max_first_energy:
                ref_max_first_energy = s.data_df.energy.iloc[0]
            else:
                pass

        fit_energy_indices = np.logical_and(
            ref_max_first_energy < unknown_spectrum.data_df.energy.values,
            unknown_spectrum.data_df.energy.values < ref_min_last_energy
        )
        log.debug('fit_energy_indices: {}'.format(fit_energy_indices))
        fit_energies = unknown_spectrum.data_df.energy.iloc[fit_energy_indices]
        log.debug('fit_energies: {}'.format(fit_energies.values))
        return fit_energies, fit_energy_indices


class FixedEnergyRangeBuilder:
    def __init__(self, energy_start, energy_stop):
        self.energy_start = energy_start
        self.energy_stop = energy_stop

    def build_range(self, unknown_spectrum, reference_spectrum_list):
        # raise exception if any of the spectra do not include the fixed energy range?
        fit_energy_indices = np.logical_and(
            self.energy_start < unknown_spectrum.data_df.energy.values,
            unknown_spectrum.data_df.energy.values < self.energy_stop
        )
        log.debug('fit_energy_indices: {}'.format(fit_energy_indices.values))
        fit_energies = unknown_spectrum.data_df.energy.iloc[fit_energy_indices]
        log.debug('fit_energies: {}'.format(fit_energies.values))
        return fit_energies, fit_energy_indices


class CombinationFitResults:
    def __init__(self, spectrum, best_fit, sorted_component_count_fit_lists):
        self.spectrum = spectrum
        self.best_fit = best_fit
        self.sorted_component_count_fit_lists = sorted_component_count_fit_lists


class AllCombinationFitTask:
    def __init__(self, energy_range_builder, reference_spectrum_list, unknown_spectrum_list, component_count_range):
        self.energy_range_builder = energy_range_builder
        self.reference_spectrum_list = reference_spectrum_list
        self.unknown_spectrum_list = unknown_spectrum_list
        self.component_count_range = component_count_range
        self.fit_table = collections.OrderedDict()

    def fit_all(self):
        log = logging.getLogger(name='fit_all')
        for unknown_spectrum in self.unknown_spectrum_list:
            log.debug('fitting {}'.format(unknown_spectrum.file_name))
            best_fit, sorted_component_count_fit_lists = self.fit(unknown_spectrum)
            self.fit_table[unknown_spectrum] = CombinationFitResults(
                spectrum=unknown_spectrum,
                best_fit=best_fit,
                sorted_component_count_fit_lists=sorted_component_count_fit_lists
            )

    def fit(self, unknown_spectrum):
        log = logging.getLogger(name=unknown_spectrum.file_name)
        # fit all combinations of references
        spectrum_fit_list = []
        for component_count in self.component_count_range:
            for reference_spectra_combination in itertools.combinations(self.reference_spectrum_list, component_count):
                log.debug('fitting to references {}'.format(reference_spectra_combination))
                fit_energies, fit_energies_ndx = self.energy_range_builder.build_range(unknown_spectrum, reference_spectra_combination)
                # interpolate the reference spectra at the fit_energies
                log.debug('fit energies.shape: {}'.format(fit_energies.shape))
                # reference_spectra_A_df is energies x components
                reference_spectra_A_series = {}
                unknown_spectrum_b = unknown_spectrum.data_df.norm[fit_energies_ndx]
                log.debug('unknown_spectrum_b.shape: {}'.format(unknown_spectrum_b.shape))
                log.debug('unknown_spectrum_b      : {}'.format(unknown_spectrum_b.values))
                log.debug('reference combination  : {}'.format(reference_spectra_combination))
                reference_spectra_A_column_list = []
                for i, rs in enumerate(reference_spectra_combination):
                    reference_spectra_A_column_list.append(rs.file_name)
                    reference_spectra_A_series[rs.file_name] = pd.Series(rs.interpolant(fit_energies))
                reference_spectra_A_df = pd.DataFrame(reference_spectra_A_series)

                # it is important to put the columns in the expected order
                reference_spectra_A_df = reference_spectra_A_df.reindex(columns=reference_spectra_A_column_list)

                ##reference_spectra_coef_x, residual, *extra = nnls(
                ##    reference_spectra_A_df.values,
                ##    unknown_spectrum_b
                ##)
                reference_spectra_coef_x, residual, rank, sigma = lstsq(
                    reference_spectra_A_df.values,
                    unknown_spectrum_b
                )
                log.debug('A        :\n{}'.format(reference_spectra_A_df))
                log.debug('coef     : {}'.format(reference_spectra_coef_x))
                log.debug('residual : {}'.format(residual))
                log.debug('solution :\n{}'.format(reference_spectra_A_df.dot(reference_spectra_coef_x)))
                if np.any(reference_spectra_coef_x < 0.0):
                    # this happens a lot with lstsq and is generally not a problem
                    #print('{}: least-squares fit has negative coefficients'.format(
                    #    unknown_spectrum_file_name
                    #))
                    continue
                ##else:
                spectrum_fit = SpectrumFit(
                    interpolant_incident_energy=fit_energies,
                    reference_spectra_A_df=reference_spectra_A_df,
                    unknown_spectrum_b=unknown_spectrum_b,
                    reference_spectra_seq=reference_spectra_combination,
                    reference_spectra_coef_x=reference_spectra_coef_x
                )
                spectrum_fit_list.append(spectrum_fit)
        best_fit, sorted_component_count_fits = self.sort_fits(unknown_spectrum, spectrum_fit_list)
        return best_fit, sorted_component_count_fits

    def sort_fits(self, spectrum, spectrum_fit_list):
        # sort all fits with the same component count together
        log = logging.getLogger(name=spectrum.file_name)
        # compare the top fits for each component count
        # component_count_fit_lists looks like this:
        #   [ [list of 0-component fits], [list of 1-component fits], ..., [list of n-component fits] ]
        component_count_fit_lists = [[] for component_count in self.component_count_range]
        # append and extra empty list for 0-component fits even though there are none
        # this allows component count to work as the list index
        component_count_fit_lists.append([])
        log.debug('creating one fit list for each component count in self.component_count_range: {}'.format(
            self.component_count_range
        ))
        log.debug('component_count_fit_lists: {}'.format(component_count_fit_lists))

        # populate the component count lists from a sorted list of all fits
        for spectrum_fit in sorted(spectrum_fit_list, key=attrgetter('nss')):
            component_count = len(spectrum_fit.reference_spectra_seq)
            component_count_fit_lists[component_count].append(spectrum_fit)

        for component_count, component_count_fit_list in enumerate(component_count_fit_lists):
            if len(component_count_fit_list) > 0:
                log.debug('best fit for {} component(s): {}'.format(component_count, component_count_fit_list[0]))
            else:
                log.debug('no fits for component count {}'.format(component_count))
        # choose the best fit from the top fits for all component counts
        # skip the 0-component list since it is empty
        best_fit = None
        previous_nss = 1.0
        for component_count, component_count_fit_list in enumerate(component_count_fit_lists[1:]):
            best_fit_for_component_count = component_count_fit_list[0]
            improvement = (previous_nss - best_fit_for_component_count.nss) / previous_nss
            log.debug('improvement: {:5.3f} for {}'.format(improvement, best_fit_for_component_count))
            if improvement < 0.10:
                break
            else:
                best_fit = best_fit_for_component_count
                previous_nss = best_fit.nss
        log.debug('best fit: {}'.format(best_fit))
        return best_fit, component_count_fit_lists

    def write_table(self, table_file_path):
        with open(table_file_path, 'w') as table_file:
            table_file.write('spectrum\tNSS')
            for spectrum, fit_results in self.fit_table.items():
                table_file.write(spectrum.file_name)
                table_file.write('\t')
                table_file.write('{:5.3f}'.format(fit_results.best_fit.nss))

    def draw_plots(self, plots_pdf_file_path):
        with PdfPages(plots_pdf_file_path) as plot_file:
            log.info('writing plots file {}'.format(plots_pdf_file_path))
            for spectrum, fit_results in self.fit_table.items():
                log.info('plotting fit for {}'.format(spectrum.file_name))
                f, ax = plt.subplots()
                ax.plot(fit_results.best_fit.interpolant_incident_energy, fit_results.best_fit.fit_spectrum_b)
                plot_file.savefig(f)

    def write_best_fit_arrays(self, best_fit_dir_path):
        for spectrum, fit_results in self.fit_table.items():
            file_base_name, file_name_ext = os.path.splitext(spectrum.file_name)
            fit_file_path = os.path.join(best_fit_dir_path, file_base_name + '_fit.txt')
            log.info('writing best fit to {}'.format(fit_file_path))

            fit_df = pd.DataFrame(
                {
                    'energy': fit_results.best_fit.interpolant_incident_energy,
                    'spectrum': fit_results.best_fit.unknown_spectrum_b,
                    'fit': fit_results.best_fit.fit_spectrum_b,
                    'residual': fit_results.best_fit.residuals}
            )
            fit_df.to_csv(fit_file_path, sep='\t', float_format='%8.4f', index=False)


    @classmethod
    def build(cls, config):
        log = logging.getLogger(name=str(cls))

        energy_range = cls.get_energy_range_from_config(config)

        # read prm
        prm_file_path = os.path.expanduser(config.get('references', 'prm'))
        log.info('reading PRM file {}'.format(prm_file_path))
        prm = PRM.read_prm(prm_file_path)
        # read reference files
        reference_spectrum_list = []
        for i, fp in enumerate(prm.reference_file_path_list):
            log.info('reading reference file {}: {}'.format(i, fp))
            reference_spectrum = ReferenceSpectrum.read_file(fp)
            reference_spectrum_list.append(reference_spectrum)

        # read data files
        unknown_spectrum_file_path_list = []
        for j, (unknown_spectrum_glob, _) in enumerate(config.items('data')):
            log.info('unknown spectrum glob: {}'.format(unknown_spectrum_glob))
            glob_pattern_expanded = os.path.expanduser(unknown_spectrum_glob)
            unknown_spectrum_file_path_list.extend(glob.glob(glob_pattern_expanded))
        log.info('found {} data files'.format(len(unknown_spectrum_file_path_list)))

        unknown_spectrum_list = []
        for unknown_spectrum_file_path in unknown_spectrum_file_path_list:
            log.info('reading data file {}'.format(unknown_spectrum_file_path))
            unknown_spectrum = Spectrum.read_file(unknown_spectrum_file_path)
            unknown_spectrum_list.append(unknown_spectrum)

        fit_task = AllCombinationFitTask(
            energy_range_builder=energy_range,
            reference_spectrum_list=reference_spectrum_list,
            unknown_spectrum_list=unknown_spectrum_list,
            component_count_range=prm.component_count_range
        )

        return fit_task

    @classmethod
    def get_energy_range_from_config(cls, config):
        if config.has_option('parameters', 'fit_energy_start') and config.has_option('parameters', 'fit_energy_stop'):
            fit_energy_start = config.getfloat('parameters', 'fit_energy_start')
            fit_energy_stop = config.getfloat('parameters', 'fit_energy_stop')
            energy_range = FixedEnergyRangeBuilder(fit_energy_start, fit_energy_stop)
        elif not(config.has_option('parameters', 'fit_energy_start') or config.has_option('parameters', 'fit_energy_stop')):
            energy_range = AdaptiveEnergyRangeBuilder()
        else:
            raise Exception('only one of fit_energy_start and fit_energy_stop was specified in the configuration')

        return energy_range
