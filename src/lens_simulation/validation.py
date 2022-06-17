
from lens_simulation.constants import LENS_SWEEPABLE_KEYS, MODIFICATION_SWEEPABLE_KEYS, BEAM_SWEEPABLE_KEYS, STAGE_SWEEPABLE_KEYS, GRATING_SWEEPABLE_KEYS, TRUNCATION_SWEEPABLE_KEYS, APERTURE_SWEEPABLE_KEYS
from lens_simulation import constants

import os
from lens_simulation import utils

import lens_simulation

DEFAULT_CONFIG = utils.load_yaml_config(os.path.join(os.path.dirname(lens_simulation.__file__), "default.yaml"))


def load_default_values(config, default_config):
    # add missing keys with default values
    for dk, dv in default_config.items():

        # add missing default values
        if dk not in config:
            config[dk] = dv

    return config

def _validate_requires_keys(config, rkeys, name: str = "") -> None:

    for rk in rkeys:

        if rk not in config:
            raise ValueError(f"Required key {rk} is not in {name} config. Required keys: {rkeys}")

def _validate_sweepable_parameters(config, sweep_keys):

    for k in sweep_keys:
        
        k_stop, k_step = f"{k}_stop", f"{k}_step"
        
        # start, stop, step
        if k not in config:
            config[k] = None

        if k_stop not in config:
            config[k_stop] = None 


        if k_step not in config:
            config[k_step] = None

    return config


def _validate_default_lens_config(lens_config: dict) -> dict:

    # required settings
    _validate_requires_keys(lens_config, constants.REQUIRED_LENS_KEYS, name="lens")

    # default settings
    lens_config = load_default_values(lens_config, DEFAULT_CONFIG["lens"])

    # validate sweepable parameters
    lens_config = _validate_sweepable_parameters(lens_config, LENS_SWEEPABLE_KEYS)

    # validate lens modifications
    lens_config = _validate_required_lens_modification_config(lens_config)
    lens_config = _validate_lens_modification_type(lens_config)

    # validate modification sweepable parameters
    if lens_config["grating"] is not None:
        lens_config["grating"] = _validate_sweepable_parameters(lens_config["grating"], GRATING_SWEEPABLE_KEYS)
    if lens_config["truncation"] is not None:
        lens_config["truncation"] = _validate_sweepable_parameters(lens_config["truncation"], TRUNCATION_SWEEPABLE_KEYS)
    if lens_config["aperture"] is not None:
        lens_config["aperture"] = _validate_sweepable_parameters(lens_config["aperture"], APERTURE_SWEEPABLE_KEYS)

    # QUERY
    # do we want to require height, diameter, exponent if the user loads a custom profile. What is required?
    # is lens_type a required parameters? how much error checking on the lens_type, e.g. if not in LensType.name etc

    return lens_config


def _validate_required_lens_modification_config(config: dict) -> dict:

    # not providing defaults, checking types

    if config["grating"] is not None:
        _validate_requires_keys(config["grating"], constants.REQUIRED_LENS_GRATING_KEYS, name="lens grating")

    if config["truncation"] is not None:
        _validate_requires_keys(config["truncation"], constants.REQUIRED_LENS_TRUNCATION_KEYS, name="lens truncation")

    if config["aperture"] is not None:
        _validate_requires_keys(config["aperture"], constants.REQUIRED_LENS_APERTURE_KEYS, name="lens aperture")

    return config


def _validate_lens_modification_type(config: dict) -> dict:

    # not providing defaults, checking types

    if config["grating"] is not None:

        # type validation
        config["grating"]["width"] = float(config["grating"]["width"])
        config["grating"]["distance"] = float(config["grating"]["distance"])
        config["grating"]["depth"] = float(config["grating"]["depth"])
        config["grating"]["x"] = bool(config["grating"]["x"])
        config["grating"]["y"] = bool(config["grating"]["y"])
        config["grating"]["centred"] = bool(config["grating"]["centred"])

    if config["truncation"] is not None:

        # type validation
        config["truncation"]["height"] = float(config["truncation"]["height"])
        config["truncation"]["radius"] = float(config["truncation"]["radius"])
        config["truncation"]["type"] = str(config["truncation"]["type"])
        config["truncation"]["aperture"] = bool(config["truncation"]["aperture"])

    if config["aperture"] is not None:

        # type validation
        config["aperture"]["inner"] = float(config["aperture"]["inner"])
        config["aperture"]["outer"] = float(config["aperture"]["outer"])
        config["aperture"]["type"] = str(config["aperture"]["type"])
        config["aperture"]["invert"] = bool(config["aperture"]["invert"])

    return config



def _validate_default_beam_config(config: dict) -> dict:
    
    # required settings
    _validate_requires_keys(config, constants.REQUIRED_BEAM_KEYS, name="beam")

    # default settings
    config["distance_mode"] = config["distance_mode"].title() if "distance_mode" in config else "Direct"
    config["spread"] = config["spread"].title() if "spread" in config else "Plane"
    config["shape"] = config["shape"].title() if "shape" in config else "Circular"


    config["n_slices"] = config["n_slices"] if "n_slices" in config else 0
    config["step_size"] = config["step_size"] if "step_size" in config else 0

    config["position_x"] = config["position_x"] if "position_x" in config else 0.0
    config["position_y"] = config["position_y"] if "position_y" in config else 0.0
    config["theta"] = config["theta"] if "theta" in config else 0.0
    config["numerical_aperture"] = config["numerical_aperture"] if "numerical_aperture" in config else None
    config["tilt_x"] = config["tilt_x"] if "tilt_x" in config else 0.0
    config["tilt_y"] = config["tilt_y"] if "tilt_y" in config else 0.0
    config["source_distance"] = config["source_distance"] if "source_distance" in config else None
    config["final_width"] = config["final_width"] if "final_width" in config else None
    config["focal_multiple"] = config["focal_multiple"] if "focal_multiple" in config else None
    config["output_medium"] = config["output_medium"] if "output_medium" in config else 1.0

    # sensible default
    if config["n_slices"]  == 0 and config["step_size"] == 0:
        config["n_slices"] = 10

    # validate the sweepable parameters
    bc = _validate_sweepable_parameters(config, BEAM_SWEEPABLE_KEYS)
    config.update(bc)

    return config

def _validate_simulation_stage_list(stages: list, simulation_lenses: dict) -> None:
    """Validate that all lenses and mediums have been defined, and all simulation stages have been
    defined correctly.
    """

    for stage in stages:
        # validate all lens, mediums exist,
        if stage["lens"] not in simulation_lenses:
            raise ValueError(f"{stage['lens']} has not been defined in the configuration")

        stage = _validate_default_simulation_stage_config(stage)

    return stages

def _validate_default_simulation_stage_config(stage_config: dict) -> dict:

    # required settings
    _validate_requires_keys(stage_config, constants.REQUIRED_SIMULATION_STAGE_KEYS, name="stage")


    # default settings
    stage_config["start_distance"] = 0 if "start_distance" not in stage_config else stage_config["start_distance"]
    stage_config["finish_distance"] = 0 if "finish_distance" not in stage_config else stage_config["finish_distance"]
    stage_config["n_slices"] = 0 if "n_slices" not in stage_config else stage_config["n_slices"]
    stage_config["step_size"] = 0 if "step_size" not in stage_config else stage_config["step_size"]
    stage_config["use_equivalent_focal_distance"] = False if "use_equivalent_focal_distance" not in stage_config else stage_config["use_equivalent_focal_distance"]
    stage_config["focal_distance_start_multiple"] = 0.0 if "focal_distance_start_multiple" not in stage_config else stage_config["focal_distance_start_multiple"]
    stage_config["focal_distance_multiple"] = 1.0 if "focal_distance_multiple" not in stage_config else stage_config["focal_distance_multiple"]

    # default conditional settings
    if stage_config["use_equivalent_focal_distance"] is True:
        stage_config["focal_distance_start_multiple"] = 0.0 if stage_config["focal_distance_start_multiple"] is None else stage_config["focal_distance_start_multiple"]
        stage_config["focal_distance_multiple"] = 1.0 if stage_config["focal_distance_multiple"] is None else stage_config["focal_distance_multiple"]

        # TODO: check the more complicated cases for these, e.g. need a height and exponent to calculate equiv focal distance

    # sensible default
    if stage_config["n_slices"] == 0 and stage_config["step_size"] == 0:
        stage_config["n_slices"] = 10

    # validate_sweepable parameters
    stage_config = _validate_sweepable_parameters(stage_config, STAGE_SWEEPABLE_KEYS)

    return stage_config


def _validate_simulation_parameters_config(config: dict) -> dict:

    _validate_requires_keys(config, constants.REQUIRED_SIMULATION_PARAMETER_KEYS, name="simulation parameters")

    return config

def _validate_simulation_options_config(config: dict) -> dict:

    # required_settings
    _validate_requires_keys(config, constants.REQUIRED_SIMULATION_OPTIONS_KEYS, name="simulation options")

    # default settings
    config["save_plot"] = True if "save_plot" not in config else config["save_plot"]
    config["save"] = False if "save" not in config else config["save"]
    config["verbose"] = False if "verbose" not in config else config["verbose"]
    config["debug"] = False if "debug" not in config else config["debug"]

    return config


def _validate_required_simulation_config(config: dict) -> dict:

    _validate_requires_keys(config, constants.REQUIRED_SIMULATION_KEYS, name = "simulation")

def _validate_simulation_config(config: dict):

    # validate required configs
    _validate_required_simulation_config(config)

    SIM_PARAMETERS_KEY = "sim_parameters"
    OPTIONS_KEY = "options"
    BEAM_KEY = "beam"
    LENS_KEY = "lenses"
    STAGE_KEY = "stages"

    # validate individual configs
    _validate_simulation_parameters_config(config[SIM_PARAMETERS_KEY])
    _validate_simulation_options_config(config[OPTIONS_KEY])
    _validate_default_beam_config(config[BEAM_KEY])

    for lens_config in config[LENS_KEY]:
        lens_config = _validate_default_lens_config(lens_config)

    for stage_config in config[STAGE_KEY]:

        _validate_default_simulation_stage_config(stage_config)

    return config