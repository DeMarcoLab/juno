import pytest

import os
from lens_simulation import validation
from lens_simulation import utils
import lens_simulation

@pytest.fixture
def config():

    config = utils.load_config(os.path.join(os.path.dirname(lens_simulation.__file__), "config.yaml"))

    return config


def test_validate_default_lens_config():

    lens_config_full = {"name": "lens_1", "diameter": 500e-6, "height": 10e-6, "exponent": 2.0, "medium": "medium_1"}

    lens_config_full = validation._validate_default_lens_config(lens_config_full)


def test_validate_default_beam_config():

    return NotImplemented
    
def test_validate_default_medium_config():

    return NotImplemented

def test_validate_default_simulation_stage_config():

    return NotImplemented

def test_validate_simulation_stage_list():

    return NotImplemented


def test_validate_simulation_config():

    return NotImplemented

def test_validate_simulation_options_config():

    return NotImplemented

def test_validate_simulation_parameters_config():

    return NotImplemented
