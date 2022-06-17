from distutils.command.config import config
from pyrsistent import CheckedKeyTypeError
import pytest

import os
from lens_simulation import validation, utils, constants
import lens_simulation

# @pytest.fixture
# def config():

#     config = utils.load_config(os.path.join(os.path.dirname(lens_simulation.__file__), "config.yaml"))

#     return config

@pytest.fixture
def valid_lens_config():

    config = dict.fromkeys(constants.REQUIRED_LENS_KEYS)
    config["grating"] = dict.fromkeys(constants.REQUIRED_LENS_GRATING_KEYS, 1)
    config["truncation"] = dict.fromkeys(constants.REQUIRED_LENS_TRUNCATION_KEYS, 1)
    config["aperture"] = dict.fromkeys(constants.REQUIRED_LENS_APERTURE_KEYS, 1)

    return config

@pytest.fixture
def invalid_lens_config():

    config = {}

    return config

@pytest.fixture
def invalid_lens_config_with_modifications():

    config = dict.fromkeys(constants.REQUIRED_LENS_KEYS)
    config["grating"] = {}
    config["truncation"] = {}
    config["aperture"] = {}

    return config

def assert_keys_are_in_config(config: dict, rkeys: list, name: str = "") -> None:

    for rk in rkeys:        
        assert rk in config, f"Required key {rk} is not in {name} config. Required keys: {rkeys}"







# TODO: 
# required, vs default values...



def test_validate_required_lens_config(valid_lens_config):

    config = valid_lens_config

    validation._validate_default_lens_config(config)

def test_validate_required_lens_config_raises_error_for_missing_values(invalid_lens_config):

    config = invalid_lens_config
    with pytest.raises(ValueError):
        validation._validate_default_lens_config(config)

def test_validate_required_lens_modification_config(valid_lens_config):

    config = valid_lens_config
    validation._validate_required_lens_modification_config(config)

def test_validate_required_lens_modification_config_raises_error_for_missing_values(invalid_lens_config_with_modifications):

    config = invalid_lens_config_with_modifications
    with pytest.raises(ValueError):
        validation._validate_required_lens_modification_config(config)


def test_validate_required_beam_config():

    config = dict.fromkeys(constants.REQUIRED_BEAM_KEYS)

    validation._validate_default_beam_config(config)

def test_validate_required_beam_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_default_beam_config(config)

    
def test_validate_required_simulation_stage_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_STAGE_KEYS)
    assert_keys_are_in_config(config, constants.REQUIRED_SIMULATION_STAGE_KEYS)

def test_validate_required_simulation_stage_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_default_simulation_stage_config(config)



# def test_validate_simulation_stage_list():

#     return NotImplemented


def test_validate_required_simulation_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_KEYS)
    validation._validate_required_simulation_config(config)


def test_validate_required_simulation_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_required_simulation_config(config)

def test_validate_simulation_options_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_OPTIONS_KEYS)

    validation._validate_simulation_options_config(config)


def test_validate_simulation_options_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_simulation_options_config(config)




def test_validate_simulation_parameters_config_passes():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_PARAMETER_KEYS)
    config = validation._validate_simulation_parameters_config(config)

    assert_keys_are_in_config(config, constants.REQUIRED_SIMULATION_PARAMETER_KEYS)

def test_validate_simulation_parameters_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        config = validation._validate_simulation_parameters_config(config)

def test_validate_sweepable_parameters():



    sweep_keys = (constants.LENS_SWEEPABLE_KEYS, constants.GRATING_SWEEPABLE_KEYS, 
                constants.TRUNCATION_SWEEPABLE_KEYS, constants.APERTURE_SWEEPABLE_KEYS, 
                constants.BEAM_SWEEPABLE_KEYS, constants.STAGE_SWEEPABLE_KEYS)

    for sk in sweep_keys:

        required_keys = []
        for k in sk:
            required_keys.append(k)
            required_keys.append(f"{k}_stop")
            required_keys.append(f"{k}_step")
        
        config = {}
        config = validation._validate_sweepable_parameters(config, sk)

        # assert keys are in config
        assert_keys_are_in_config(config, sk)
