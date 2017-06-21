"""
The MIT License (MIT)

Copyright (c) 2015-2017 Joshua Lynch, Sarah Nicholas

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
import logging

import pytest

import mrfitty.fit_task_builder as fit_task_builder


_spectrum_file_content = """\
  11825.550       0.62757215E-02   0.62429776E-02  -0.58947170E-03
  11830.550       0.30263933E-02   0.29936416E-02  -0.55479576E-03
  11835.550       0.15935143E-03   0.12659210E-03  -0.45673882E-03
  11840.550      -0.20089439E-02  -0.20417109E-02  -0.31491527E-03
"""


def test__get_required_config_value():
    config = fit_task_builder.get_config_parser()

    test_section = 'blah'
    test_option = 'bleh'
    test_value = 'blih'

    config.add_section(section=test_section)
    config.set(section=test_section, option=test_option, value=test_value)

    assert test_value == fit_task_builder._get_required_config_value(
        config=config, section=test_section, option=test_option)

    with pytest.raises(fit_task_builder.ConfigurationFileError):
        fit_task_builder._get_required_config_value(config=config, section='missing', option='missing')


def test_build_reference_spectrum_list_from_prm_section(fs):
    reference_config = fit_task_builder.get_config_parser()
    reference_config.read_string("""\
[prm]
NBCompoMax = 4
NBCompoMin = 1
arsenate_aqueous_avg_als_cal.e
arsenate_sorbed_anth_avg_als_cal.e
""")

    fs.CreateFile(file_path='arsenate_aqueous_avg_als_cal.e', contents=_spectrum_file_content)
    fs.CreateFile(file_path='arsenate_sorbed_anth_avg_als_cal.e', contents=_spectrum_file_content)

    max_count, min_count, reference_list = \
        fit_task_builder.build_reference_spectrum_list_from_config_prm_section(reference_config)

    assert max_count == 4
    assert min_count == 1
    assert len(reference_list) == 2


def test_build_reference_spectrum_list_from_prm_section__bad_component_counts(fs):
    reference_config = fit_task_builder.get_config_parser()
    reference_config.read_string("""\
[prm]
NBCompoMax = 1
NBCompoMin = 4
arsenate_aqueous_avg_als_cal.e
arsenate_sorbed_anth_avg_als_cal.e
""")

    fs.CreateFile(file_path='arsenate_aqueous_avg_als_cal.e', contents=_spectrum_file_content)
    fs.CreateFile(file_path='arsenate_sorbed_anth_avg_als_cal.e', contents=_spectrum_file_content)

    with pytest.raises(fit_task_builder.ConfigurationFileError):
        fit_task_builder.build_reference_spectrum_list_from_config_prm_section(reference_config)


def test_build_reference_spectrum_list_from_config_file(fs):
    reference_config = fit_task_builder.get_config_parser()
    reference_config.read_string("""\
[references]
references/*.e    
""")

    fs.CreateDirectory(directory_path='references')
    fs.CreateFile(file_path='references/arsenate_aqueous_avg_als_cal.e', contents=_spectrum_file_content)
    fs.CreateFile(file_path='references/arsenate_sorbed_anth_avg_als_cal.e', contents=_spectrum_file_content)

    reference_list = fit_task_builder.build_reference_spectrum_list_from_config_file(reference_config)

    assert len(reference_list) == 2


def test_build_unknown_spectrum_list_from_config_file(fs):
    data_config = fit_task_builder.get_config_parser()
    data_config.read_string("""\
[data]
data/*.e
""")

    fs.CreateDirectory(directory_path='data')
    fs.CreateFile(file_path='data/data_0.e', contents=_spectrum_file_content)
    fs.CreateFile(file_path='data/data_1.e', contents=_spectrum_file_content)

    data_list = fit_task_builder.build_unknown_spectrum_list_from_config_file(data_config)

    assert len(data_list) == 2


def test_get_fit_parameters_from_config_file():
    fit_config = fit_task_builder.get_config_parser()
    fit_config.read_string("""\
[fit]    
max_component_count = 3
min_component_count = 1

""")