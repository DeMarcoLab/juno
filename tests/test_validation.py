import pytest
from star_glass import validation, utils, constants

DEFAULT_CONFIG = validation.get_default_config()

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

def assert_config_has_default_values(config: dict, default_key: str) -> None: 
    for dk, dv in validation.DEFAULT_CONFIG[default_key].items():

        if  isinstance(config[dk], dict):
            pass # we dont do default values for nested modifications...
        else:
            assert config[dk]  == dv, f"Non-default value found for key {dk}, {config[dk]} should be {dv}"
        
## UTILITIES
def test_load_default_values():

    default_config = {"key1": 1, "key2": 2, "key3": 3}
    config = {"key1": 0}

    config = validation.load_default_values(config, default_config)

    assert config["key1"] == 0, "Default value should not be written"
    assert config["key2"] == default_config["key2"], "Default value should be written"
    assert config["key3"] == default_config["key3"], "Default value should be written"


def test_validate_required_keys():
    
    config = {"key1": 1, "key2": 2, "key3": 3}
    config_missing = {"key1": 1, "key2": 2}
    required_keys = ["key1", "key2", "key3"]
    
    # keys in
    validation._validate_required_keys(config, required_keys)

    # keys not in
    with pytest.raises(ValueError):
        validation._validate_required_keys(config_missing, required_keys)

## LENS
def test_validate_required_lens_config(valid_lens_config):

    config = valid_lens_config

    validation._validate_default_lens_config(config)

def test_validate_required_lens_config_raises_error_for_missing_values(invalid_lens_config):

    config = invalid_lens_config
    with pytest.raises(ValueError):
        validation._validate_default_lens_config(config)


def test_default_lens_config_values(valid_lens_config):
    # a valid but empty config, should have default values...
    config = valid_lens_config

    config = validation._validate_default_lens_config(config)

    assert_config_has_default_values(config, default_key="lens")


def test_validate_required_lens_modification_config(valid_lens_config):

    config = valid_lens_config
    validation._validate_required_lens_modification_config(config)

def test_validate_required_lens_modification_config_raises_error_for_missing_values(invalid_lens_config_with_modifications):

    config = invalid_lens_config_with_modifications
    with pytest.raises(ValueError):
        validation._validate_required_lens_modification_config(config)

## BEAM
def test_validate_required_beam_config():

    config = dict.fromkeys(constants.REQUIRED_BEAM_KEYS)

    validation._validate_default_beam_config(config)

def test_validate_required_beam_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_default_beam_config(config)

def test_validate_default_beam_config():

    config = dict.fromkeys(constants.REQUIRED_BEAM_KEYS)

    config = validation._validate_default_beam_config(config)

    assert_config_has_default_values(config, "beam")


def test_validatie_beam_config_with_case():

    config = dict.fromkeys(constants.REQUIRED_BEAM_KEYS)

    config["shape"] = "rectangular"
    config["spread"] = "plane"
    config["distance_mode"] = "direct"

    config = validation._validate_default_beam_config(config)

    assert config["shape"] == config["shape"].title()
    assert config["spread"] == config["spread"].title()
    assert config["distance_mode"] == config["distance_mode"].title()



## STAGE
def test_validate_required_simulation_stage_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_STAGE_KEYS)
    assert_keys_are_in_config(config, constants.REQUIRED_SIMULATION_STAGE_KEYS)

def test_validate_required_simulation_stage_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_default_simulation_stage_config(config)

def test_validate_default_stage_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_STAGE_KEYS)

    config = validation._validate_default_simulation_stage_config(config)

    assert_config_has_default_values(config, "stage")

## SIMULATION
def test_validate_required_simulation_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_KEYS)
    validation._validate_required_simulation_config(config)

def test_validate_required_simulation_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_required_simulation_config(config)


## SIMULATION OPTIONS
def test_validate_simulation_options_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_OPTIONS_KEYS)

    validation._validate_simulation_options_config(config)


def test_validate_simulation_options_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        validation._validate_simulation_options_config(config)

def test_validate_default_simulation_options_config():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_OPTIONS_KEYS)

    config = validation._validate_simulation_options_config(config)

    assert_config_has_default_values(config, "options")

## SIMULATION PARAMETERS
def test_validate_simulation_parameters_config_passes():

    config = dict.fromkeys(constants.REQUIRED_SIMULATION_PARAMETER_KEYS)
    config = validation._validate_simulation_parameters_config(config)

    assert_keys_are_in_config(config, constants.REQUIRED_SIMULATION_PARAMETER_KEYS)

def test_validate_simulation_parameters_config_raises_error_for_missing_values():

    config = {}
    with pytest.raises(ValueError):
        config = validation._validate_simulation_parameters_config(config)



## SWEEPABLE PARAMETERS
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

def test_validate_sweep_parameters_doesnt_override_existing_values():


    config = dict.fromkeys(constants.LENS_SWEEPABLE_KEYS, 1)

    config = validation._validate_sweepable_parameters(config, constants.LENS_SWEEPABLE_KEYS)

    for k in constants.LENS_SWEEPABLE_KEYS:

        assert config[k] == 1, f"Validating parameter sweep has changed existing value for {k}. {config[k]} should be 1."