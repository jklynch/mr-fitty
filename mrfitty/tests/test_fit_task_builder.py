import pytest

from mrfitty.fit_task_builder import (
    build_reference_spectrum_list_from_config_file,
    build_reference_spectrum_list_from_config_prm_section,
    build_unknown_spectrum_list_from_config_file,
    ConfigurationFileError,
    get_config_parser,
    get_fit_parameters_from_config_file,
    _get_required_config_value,
)

_spectrum_file_content = """\
  11825.550       0.62757215E-02   0.62429776E-02  -0.58947170E-03
  11830.550       0.30263933E-02   0.29936416E-02  -0.55479576E-03
  11835.550       0.15935143E-03   0.12659210E-03  -0.45673882E-03
  11840.550      -0.20089439E-02  -0.20417109E-02  -0.31491527E-03
"""


def test__get_required_config_value():
    config = get_config_parser()

    test_section = "blah"
    test_option = "bleh"
    test_value = "blih"

    config.add_section(section=test_section)
    config.set(section=test_section, option=test_option, value=test_value)

    assert test_value == _get_required_config_value(
        config=config, section=test_section, option=test_option
    )

    with pytest.raises(ConfigurationFileError):
        _get_required_config_value(config=config, section="missing", option="missing")


def test_build_reference_spectrum_list_from_prm_section(fs):
    reference_config = get_config_parser()
    reference_config.read_string(
        """\
[prm]
NBCompoMax = 4
NBCompoMin = 1
arsenate_aqueous_avg_als_cal.e
arsenate_sorbed_anth_avg_als_cal.e
"""
    )

    fs.create_file(
        file_path="arsenate_aqueous_avg_als_cal.e", contents=_spectrum_file_content
    )
    fs.create_file(
        file_path="arsenate_sorbed_anth_avg_als_cal.e", contents=_spectrum_file_content
    )

    (
        max_count,
        min_count,
        reference_list,
    ) = build_reference_spectrum_list_from_config_prm_section(reference_config)

    assert max_count == 4
    assert min_count == 1
    assert len(reference_list) == 2


def test_build_reference_spectrum_list_from_prm_section__bad_component_counts(fs):
    reference_config = get_config_parser()
    reference_config.read_string(
        """\
[prm]
NBCompoMax = 1
NBCompoMin = 4
arsenate_aqueous_avg_als_cal.e
arsenate_sorbed_anth_avg_als_cal.e
"""
    )

    fs.create_file(
        file_path="arsenate_aqueous_avg_als_cal.e", contents=_spectrum_file_content
    )
    fs.create_file(
        file_path="arsenate_sorbed_anth_avg_als_cal.e", contents=_spectrum_file_content
    )

    with pytest.raises(ConfigurationFileError):
        build_reference_spectrum_list_from_config_prm_section(reference_config)


def test_build_reference_spectrum_list_from_config_file(fs):
    reference_config = get_config_parser()
    reference_config.read_string(
        """\
[references]
references/*.e
"""
    )

    fs.create_dir(directory_path="references")
    fs.create_file(
        file_path="references/arsenate_aqueous_avg_als_cal.e",
        contents=_spectrum_file_content,
    )
    fs.create_file(
        file_path="references/arsenate_sorbed_anth_avg_als_cal.e",
        contents=_spectrum_file_content,
    )

    reference_list = build_reference_spectrum_list_from_config_file(reference_config)

    assert len(reference_list) == 2


def test_build_unknown_spectrum_list_from_config_file(fs):
    data_config = get_config_parser()
    data_config.read_string(
        """\
[data]
data/*.e
"""
    )

    fs.create_dir(directory_path="data")
    fs.create_file(file_path="data/data_0.e", contents=_spectrum_file_content)
    fs.create_file(file_path="data/data_1.e", contents=_spectrum_file_content)

    data_list = build_unknown_spectrum_list_from_config_file(data_config)

    assert len(data_list) == 2


def test_get_fit_parameters_from_config_file():
    fit_config = get_config_parser()
    fit_config.read_string(
        """\
[fit]
max_component_count = 3
min_component_count = 1
"""
    )

    (
        max_cmp,
        min_cmp,
        fit_method_class,
        fit_task_class,
        bootstrap_count,
    ) = get_fit_parameters_from_config_file(fit_config, prm_max_cmp=3, prm_min_cmp=1)
    assert max_cmp == 3
    assert min_cmp == 1
    assert bootstrap_count == 1000


def test_get_good_fit_parameter_bootstrap_count_from_config_file():
    fit_config = get_config_parser()
    fit_config.read_string(
        """\
[fit]
max_component_count = 3
min_component_count = 1
bootstrap_count = 2000
"""
    )

    (
        max_cmp,
        min_cmp,
        fit_method_class,
        fit_task_class,
        bootstrap_count,
    ) = get_fit_parameters_from_config_file(fit_config, prm_max_cmp=3, prm_min_cmp=1)
    assert max_cmp == 3
    assert min_cmp == 1
    assert bootstrap_count == 2000


def test_get_bad_fit_parameter_bootstrap_count_from_config_file():
    fit_config = get_config_parser()
    fit_config.read_string(
        """\
[fit]
max_component_count = 3
min_component_count = 1
bootstrap_count = haha
"""
    )

    with pytest.raises(ConfigurationFileError):
        (
            max_cmp,
            min_cmp,
            fit_method_class,
            fit_task_class,
            bootstrap_count,
        ) = get_fit_parameters_from_config_file(
            fit_config, prm_max_cmp=3, prm_min_cmp=1
        )
