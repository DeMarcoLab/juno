import itertools
import os
import sys
import numpy as np

import utils
import yaml
from juno import SimulationRunner, validation
from copy import deepcopy

from juno_custom.tools import element_tools
from juno import utils as j_utils

def generate_beam_config(config: dict):
    if config.get("beam") is None or config["beam"].get("match"):
        config = match_beam(config=config)

    return config


def match_beam(config: dict):
    # TODO: check for spherical here?
    beam_config = deepcopy(config.get("beam"))
    beam_config["beam_type"] = config.get("beam_type")
    beam_config["height"] = config.get("length")
    beam_config["width"] = config.get("diameter")
    beam_config["distance_mode"] = "Direct"
    beam_config["source_distance"] = 1.e-9
    beam_config["n_steps"] = 2
    beam_config["numerical_aperture"] = 0.0
    beam_config["position_x"] = 0.0
    beam_config["position_y"] = 0.0
    beam_config["shape"] = config.get("lens_type")
    beam_config["spread"] = "Plane"
    beam_config["step_size"] = 0.0
    beam_config["tilt_x"] = 0.0
    beam_config["tilt_y"] = 0.0
    beam_config["theta"] = 0.0

    config["beam"] = beam_config
    return config
