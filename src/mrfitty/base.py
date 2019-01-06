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
import glob
import logging
import os.path
import re

import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class Spectrum:
    """Spectrum

    Encapsulates spectrum data from a file.

    Attributes
    ----------
    file_path     : str
    file_name     : str
    data_df       : pandas.Dataframe
        Array of incident energies and fluorescence, e.g.
                           norm
        energy
        1000.1             0.001
        1000.2             0.002
        ...                ...
        1100.3             0.100
    header_fields : dictionary of header fields from spectrum file
    """
    def __init__(self, file_path, data_df, header_fields):
        self.file_path = file_path
        self.file_name = os.path.split(file_path)[1]
        self.data_df = data_df
        self.header_fields = header_fields

    def __repr__(self):
        return 'Spectrum({}, {})'.format(self.file_path, self.data_df.shape)

    @classmethod
    def read_header(cls, file_path_or_buffer):
        """
        Determine if file_path_or_buffer is a file path or a buffer. If passed a file path open the file. Read the
        input in either case. If passed a file-like object rewind it so it can be read again for energy and fluorescence.

        Parameters
        ----------
        file_path_or_buffer : str or file-like object

        Returns
        -------
        dictionary of header fields and values
        """
        if isinstance(file_path_or_buffer, str):
            with open(file_path_or_buffer, 'rt') as f:
                return cls.read_header_(f)
        else:
            header_fields = cls.read_header_(file_path_or_buffer)
            file_path_or_buffer.seek(0)
            return header_fields

    @classmethod
    def read_header_(cls, file_):
        """
        Parse header fields.

        Parameters
        ----------
        file_ : a file-like object

        Returns
        -------
        dictionary of header fields and values
        """
        header_fields = {}
        header_field_pattern = re.compile(r"^#\s*(?P<field>[^:]+)\s*:\s*(?P<value>.+)\s*$")
        for line in file_:
            m = header_field_pattern.match(line.strip())
            if m is not None:
                header_fields[m.group('field')] = m.group('value')
            else:
                pass
        return header_fields

    @classmethod
    def read_file(cls, file_path_or_buffer, **kwargs):
        """Read a spectrum file.

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
        log = logging.getLogger(name=cls.__name__)

        header_fields = cls.read_header(file_path_or_buffer)

        spectrum_data_df = pd.read_table(
            file_path_or_buffer,
            engine='python',
            sep='[ \t]+',
            comment='#',
            header=None,
            index_col=0
        )

        log.info('read %s', file_path_or_buffer)

        log.debug('  shape is %s', spectrum_data_df.shape)
        if spectrum_data_df.shape[1] < 1:
            raise Exception('{} has fewer than 2 columns'.format(file_path_or_buffer))

        # keep only the first two columns
        # the first column (energy) is the DataFrame index
        spectrum_data_df = spectrum_data_df.iloc[:, :1]
        # assign names to the index and column
        spectrum_data_df.index.rename('energy', inplace=True)
        spectrum_data_df.columns = ['norm']

        if np.isnan(spectrum_data_df.values).any():
            raise Exception('{} has one or more NaN values'.format(file_path_or_buffer))

        log.debug('  first incident energy is %d', spectrum_data_df.index[0])
        log.debug('  last incident energy is  %d', spectrum_data_df.index[-1])

        if isinstance(file_path_or_buffer, str):
            file_path = file_path_or_buffer
        else:
            file_path = 'in-memory'

        return cls(file_path, spectrum_data_df, header_fields, **kwargs)

    @classmethod
    def read_all(cls, file_glob_list):
        """
        Parse all spectrum files defined in a list of file paths and/or file globs.
        '~' will be expanded to the user's home directory.

        Parameters
        ----------
        file_glob_list : list of file paths and/or file globs

        Returns
        -------
        Set of Spectrum or Spectrum subclass instances
        Set of duplicate file paths
        """
        log = logging.getLogger(name=cls.__name__)

        # keep a list of config file entries for error reporting
        spectrum_file_path_set = set()

        # return this set of duplicate file paths
        duplicate_file_path_set = set()
        # return this set of Spectrum instances
        spectrum_set = set()
        for spectrum_file_glob in file_glob_list:
            log.info('file pattern: {}'.format(spectrum_file_glob))
            spectrum_file_glob_expanded = os.path.expanduser(spectrum_file_glob)
            log.info('expanded file pattern: {}'.format(spectrum_file_glob_expanded))
            for i, spectrum_file_path in enumerate(glob.glob(spectrum_file_glob_expanded)):
                if spectrum_file_path in spectrum_file_path_set:
                    log.info('  reference file {} has already been read'.format(spectrum_file_path))
                    duplicate_file_path_set.add(spectrum_file_path)
                else:
                    log.info('  reading reference file {}: {}'.format(i, spectrum_file_path))
                    spectrum = cls.read_file(spectrum_file_path)
                    spectrum_set.add(spectrum)

        return spectrum_set, duplicate_file_path_set


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
    def __init__(self, file_path, reference_spectrum_data, header_fields, mineral_category=None):
        super().__init__(file_path, reference_spectrum_data, header_fields)
        self.mineral_category = mineral_category
        self.interpolant = InterpolatedUnivariateSpline(
            reference_spectrum_data.index.values,
            reference_spectrum_data.norm.values
        )

        if 'Valence group' in self.header_fields:
            self.file_name = '{} [{}]'.format(self.file_name, self.header_fields['Valence group'])

    def __repr__(self):
        return 'ReferenceSpectrum({}, {}, {})'.format(self.file_path, self.data_df.shape, self.mineral_category)


class InterpolatedSpectrumSet:
    def __init__(self, spectrum_set):
        self.interpolated_set_df = InterpolatedSpectrumSet.get_interpolated_spectrum_set_df(
            spectrum_set=spectrum_set)

    @staticmethod
    def get_interpolated_spectrum_set_df(energy_range, spectrum_set):
        """ Return a pandas.DataFrame of spectrum values interpolated at the specified energies.


        """
        # the interpolated spectra will be len(energy_range) x len(spectrum_set)
        interpolated_spectra = np.zeros((len(energy_range), len(spectrum_set)))
        column_names = []
        for i, spectrum in enumerate(sorted(list(spectrum_set), key=lambda s: s.file_name)):
            column_names.append(spectrum.file_name)
            interpolated_spectra[:, i] = spectrum.interpolant(energy_range)
            # set values that would be extrapolated to NaN
            ndx = InterpolatedSpectrumSet.get_extrapolated_value_index(
                interpolated_energy=energy_range,
                measured_energy=spectrum.data_df.index.values)
            # print(ndx)
            interpolated_spectra[ndx, i] = np.nan

        interpolated_spectra_df = pd.DataFrame(
            data=interpolated_spectra,
            index=energy_range,
            columns=column_names)

        return interpolated_spectra_df

    @staticmethod
    def get_extrapolated_value_index(interpolated_energy, measured_energy):
        """Return a boolean array with True indicating interpolated energies outside the measured energy range.

        :param interpolated_energy (np.array)
        :param measured_energy (np.array)
        :returns (numpy boolean array) 
        """
        extrapolated_value_boolean_index = np.logical_or(
            interpolated_energy < measured_energy[0],
            interpolated_energy > measured_energy[-1])
        return np.where(extrapolated_value_boolean_index)


class InterpolatedReferenceSpectraSet:
    """
    An instance of this class is constructed with the full reference set for fitting the unknown spectra. The method
    get_reference_subset_and_unknown_df(...) is then used to extract a subset of the full reference set to be used for
    a single fit. For example the full reference set could have 30 spectra and get_reference_subset_and_unknown_df()
    would be called for each 1-, 2-, 3-element subset of the reference set.
    """
    def __init__(self, unknown_spectrum, reference_set):
        self.unknown_spectrum = unknown_spectrum
        self.reference_set = reference_set
        self.reference_table = {r.file_path: r for r in reference_set}
        self.interpolated_reference_set_df = InterpolatedReferenceSpectraSet.get_interpolated_reference_set_df(
            unknown_spectrum=unknown_spectrum,
            reference_set=reference_set)

    @staticmethod
    def get_interpolated_reference_set_df(unknown_spectrum, reference_set):
        """
        Given an unknown spectrum and set of reference spectra return a single DataFrame of reference fluoresence
        values interpolated at each energy value of the unknown. There may be some missing values represented by NaN.

        For example, given unknown and references:
          unknown        ref_1         ref_2         ref_3
          energy fluor   energy fluor  energy fluor  energy fluor
          100.0  50       99.0  50     101.0  50     103.0  50
          102.0  60      101.0  60     103.0  60     104.0  60
          104.0  70      102.0  70     105.0  70     105.0  70
          106.0  80      104.0  80     107.0  80     106.0  80

        return a DataFrame like this:
          energy ref_1 ref_2 ref3
          100.0   55   NaN   NaN
          102.0   70    55   NaN
          104.0   80    65    60
          106.0  NaN    75    80

        Parameters
        ----------
        unknown_spectrum - instance of Spectrum
        reference_set    - list of ReferenceSpectrum

        Returns
        -------
        DataFrame with shape (unknown_spectrum.data_df.shape[0], len(reference_set))

        """
        # the interpolated reference spectra will be unknown_spectrum.data_df.shape[0] x len(reference_set)
        interpolated_reference_spectra = np.zeros((unknown_spectrum.data_df.shape[0], len(reference_set)))
        column_names = []
        for i, reference_spectrum in enumerate(sorted(list(reference_set), key=lambda r: r.file_name)):
            column_names.append(reference_spectrum.file_name)
            interpolated_reference_spectra[:, i] = reference_spectrum.interpolant(
                unknown_spectrum.data_df.index.values
            )
            ndx = InterpolatedReferenceSpectraSet.get_extrapolated_value_index(
                unknown_energy=unknown_spectrum.data_df.index.values,
                reference_energy=reference_spectrum.data_df.index.values
            )
            #print(ndx)
            interpolated_reference_spectra[ndx, i] = np.nan

        interpolated_reference_spectra_df = pd.DataFrame(
            data=interpolated_reference_spectra,
            index=unknown_spectrum.data_df.index,
            columns=column_names)
        interpolated_reference_spectra_df.rename_axis('energy', inplace=True)

        return interpolated_reference_spectra_df

    @staticmethod
    def get_extrapolated_value_index(unknown_energy, reference_energy):
        extrapolated_value_boolean_index = np.logical_or(
            unknown_energy < reference_energy[0],
            unknown_energy > reference_energy[-1]
        )
        return np.where(extrapolated_value_boolean_index)

    #@profile
    def get_reference_subset_and_unknown_df(self, reference_list, energy_range_builder):
        log = logging.getLogger(name=self.__class__.__name__)

        reference_name_list = sorted([r.file_name for r in reference_list])
        keep_rows = self.interpolated_reference_set_df.loc[:, reference_name_list].notnull().all(axis=1)
        reference_subset_df = self.interpolated_reference_set_df.loc[keep_rows.values, reference_name_list]
        unknown_subset_df = self.unknown_spectrum.data_df.loc[reference_subset_df.index]

        fit_energies, fit_energy_indices = energy_range_builder.build_range(
            unknown_spectrum=self.unknown_spectrum,
            reference_spectrum_seq=reference_list)

        # TODO: go back to the beginning and index on energy
        #log.info('reference_subset_df:\n%s', reference_subset_df.head())
        #for r in reference_list:
        #    log.info(
        #        'reference_subset_df.loc[fit_energies, (%s, )]:\n%s',
        #        r.file_name,
        #        reference_subset_df.loc[fit_energies, (r.file_name, )].head()
        #    )

        rs = []
        for r in reference_list:
            df = reference_subset_df.loc[fit_energies, (r.file_name,)]
            df.columns = ['norm']
            rs.append(
                ReferenceSpectrum(
                    file_path=r.file_path,
                    reference_spectrum_data=df,
                    header_fields=r.header_fields,
                    mineral_category=r.mineral_category
                )
            )

        return {
            'reference_subset_df': reference_subset_df.loc[fit_energies],
            'unknown_subset_df': unknown_subset_df.loc[fit_energies],
            # note use of (r.file_name, ) to force a DataFrame to be returned
            'reference_subset_spectra': tuple(rs),
            'unknown_subset_spectrum': Spectrum(
                file_path=self.unknown_spectrum.file_path,
                data_df=unknown_subset_df.loc[fit_energies],
                header_fields=self.unknown_spectrum.header_fields
            )
        }

        #return reference_subset_df.loc[fit_energies], unknown_subset_df.loc[fit_energies]


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
    def __init__(
        self,
        interpolant_incident_energy,
        reference_spectra_A_df,
        #unknown_spectrum_b,
        unknown_spectrum,
        reference_spectra_seq,
        reference_spectra_coef_x
    ):
        log = logging.getLogger(name=self.__class__.__name__)
        self.interpolant_incident_energy = interpolant_incident_energy
        self.reference_spectra_seq = reference_spectra_seq
        self.reference_spectra_A_df = reference_spectra_A_df
        # TODO: fix this
        # TODO: remember what needs fixing
        # TODO: now I remember: make '.norm' configurable
        self.unknown_spectrum = unknown_spectrum
        #self.unknown_spectrum_b = unknown_spectrum_b.norm
        self.unknown_spectrum_b = unknown_spectrum.data_df.norm
        self.reference_spectra_coef_x = reference_spectra_coef_x
        self.fit_spectrum_b = reference_spectra_A_df.dot(reference_spectra_coef_x)
        self.residuals = self.fit_spectrum_b - self.unknown_spectrum_b
        #log.debug('self.unknown_spectrum_b :\n%s', self.unknown_spectrum_b)
        #log.debug('self.fit_spectrum_b     :\n%s', self.fit_spectrum_b)
        #log.debug('residuals               :\n%s', self.residuals)

        self.sum_of_abs_residuals = np.sum(np.abs(self.residuals))
        self.sum_of_abs_unknown_spectrum_b = np.sum(np.abs(self.unknown_spectrum_b))
        self.sum_of_squared_residuals = np.sum(np.power(self.residuals, 2.0))
        self.sum_of_squared_unknown_spectrum_b = np.sum(np.power(self.unknown_spectrum_b, 2.0))

        self.nsa = self.sum_of_abs_residuals / self.sum_of_abs_unknown_spectrum_b
        self.nss = self.sum_of_squared_residuals / self.sum_of_squared_unknown_spectrum_b

        self.reference_contribution_percent_sr = None
        self.reference_only_contribution_percent_sr = None
        self.total_reference_contribution = None
        self.residuals_contribution = None

        # calculate the approximate area under each curve
        self.residuals_auc = self.sum_of_abs_residuals
        self.unknown_spectrum_auc = self.sum_of_abs_unknown_spectrum_b

    def get_start_energy(self):
        return self.unknown_spectrum.data_df.index[0]

    def get_end_energy(self):
        return self.unknown_spectrum.data_df.index[-1]

    def get_reference_contributions_sr(self):
        log = logging.getLogger(name=self.__class__.__name__)
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

    def get_reference_only_contributions_sr(self):
        # calculate reference-only contributions as well
        # scaled_references_abs_sums_sr looks like this:
        #   Arsenopyrite_Julcani_OA.e           0.508400
        #   arsenate_sorbed_diop_avg_als_cal.e  0.092146
        #   orpiment_all_ref_als_cal.e          0.399454
        #   dtype: float64
        scaled_references_df = self.reference_spectra_coef_x * self.reference_spectra_A_df
        scaled_references_abs_sums_sr = scaled_references_df.abs().sum()

        self.reference_only_contribution_percent_sr = \
            100.0 * scaled_references_abs_sums_sr / scaled_references_abs_sums_sr.sum()
        return self.reference_only_contribution_percent_sr

    def __str__(self):
        contribution_str = 'SpectrumFit: {:5.3f}: NSS'.format(self.nss)
        reference_contribution_sr = self.get_reference_contributions_sr()
        for reference_spectrum in self.reference_spectra_seq:
            # TODO: something smarter than reference_spectrum.file_name
            reference_contribution = reference_contribution_sr[reference_spectrum.file_name]
            contribution_str += ' {:4.2f}: {},'.format(reference_contribution, reference_spectrum.file_name)
        contribution_str += ' {:4.2f}: residuals'.format(self.residuals_contribution)
        return contribution_str


class AdaptiveEnergyRangeBuilder:
    """AdaptiveEnergyRangeBuilder

    Builds an array of incident energies present in the specified unknown spectrum and the sequence of reference
    spectra.  For example, given an unknown spectrum like this:

    index             0        1        2        3
    incident energy   11760.0  11765.0  11771.0  11776.0
    fluorescence      0.08123  0.08234  0.08345  0.08456

    and two reference spectra like these:

    index             0        1        2        3
    incident energy   11761.0  11764.0  11770.0  11775.0
    fluorescence      0.08123  0.08234  0.08345  0.08456

    index             0        1        2        3
    incident energy   11759.0  11766.0  11772.0  11778.0
    fluorescence      0.08123  0.08234  0.08345  0.08456

    the incident energies present in the unknown spectrum that will be used to interpolate values from the reference
    spectra are:

    index
    incident energy   11765.0  11771.0
    """
    def __init__(self):
        pass

    #@profile
    def build_range(self, unknown_spectrum, reference_spectrum_seq):
        """

        :param unknown_spectrum:
        :param reference_spectrum_seq:
        :return:
        """
        log = logging.getLogger(name=self.__class__.__name__)
        ref_min_last_energy = np.inf
        ref_max_first_energy = -1.0 * np.inf
        for reference_spectrum in reference_spectrum_seq:
            log.debug('s: %s', reference_spectrum)
            if reference_spectrum.data_df.index[-1] < ref_min_last_energy:
                ref_min_last_energy = reference_spectrum.data_df.index[-1]
            else:
                pass
            if reference_spectrum.data_df.index[0] > ref_max_first_energy:
                ref_max_first_energy = reference_spectrum.data_df.index[0]
            else:
                pass

        fit_energy_indices = np.logical_and(
            ref_max_first_energy < unknown_spectrum.data_df.index.values,
            unknown_spectrum.data_df.index.values < ref_min_last_energy
        )
        #log.debug('fit_energy_indices: %s', fit_energy_indices)
        fit_energies = unknown_spectrum.data_df.index[fit_energy_indices]
        #log.debug('fit_energies: %s', fit_energies.values)
        return fit_energies, fit_energy_indices


class FixedEnergyRangeBuilder:
    def __init__(self, energy_start, energy_stop):
        self.energy_start = energy_start
        self.energy_stop = energy_stop

    def build_range(self, unknown_spectrum, reference_spectrum_seq):
        log = logging.getLogger(name=self.__class__.__name__)
        # raise exception if any of the reference spectra do not include the fixed energy range?
        fit_energy_indices = np.logical_and(
            self.energy_start <= unknown_spectrum.data_df.index.values,
            unknown_spectrum.data_df.index.values <= self.energy_stop
        )
        log.debug('fit_energy_indices: %s', fit_energy_indices)
        fit_energies = unknown_spectrum.data_df.index[fit_energy_indices]
        log.debug('fit_energies: %s', fit_energies.values)
        for reference_spectrum in reference_spectrum_seq:
            if reference_spectrum.data_df.index[0] > fit_energies[0]:
                raise Exception('reference spectrum {} lowest energy is above the specified start of {}'.format(
                    reference_spectrum.file_name, self.energy_start
                ))
            else:
                pass

            if reference_spectrum.data_df.index[-1] < fit_energies[-1]:
                raise Exception('refrence spectrum {} highest energy is below the specified stop of {}'.format(
                    reference_spectrum.file_name, self.energy_stop
                ))
            else:
                pass

        return fit_energies, fit_energy_indices


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
        log = logging.getLogger(name=self.__class__.__name__)
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
        log = logging.getLogger(name=cls.__name__)
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
