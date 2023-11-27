import itertools
import os
import sys
import numpy as np

import utils
import yaml
from juno import SimulationRunner, validation

from juno_custom.tools import element_tools, beam_tools
from juno_custom import main
from juno import utils as j_utils
from copy import deepcopy


def get_parameter_combinations(config: dict, key_list: list):
    sweep_keys = key_list
    config = validation._validate_sweepable_parameters(config, sweep_keys)
    kp = SimulationRunner.sweep_config_keys(config, sweep_keys)
    param_combos = list(itertools.product(*kp))
    print(param_combos)
    return param_combos


def iterate_through_parameter_combinations(config: dict, key_list: list):
    param_combos = get_parameter_combinations(config=config, key_list=key_list)
    # for i, param_combo in enumerate(param_combos):
    #     for j, key in enumerate(config["keys"]):
    #         if isinstance(param_combo[j], float):
    #             config[key] = float(param_combo[j])
    #         elif isinstance(param_combo[j], str):
    #             config[key] = str(param_combo[j])

        # main.run(config=config, count=i)


def save_outputs(config: dict, profile: np.ndarray, count=0):
    save_config(
        config,
        os.path.join(
            config["system"].get("configs_path"), "config_" + str(count) + ".yaml"
        ),
    )
    save_profile(
        profile=profile,
        filename=os.path.join(
            config["system"].get("configs_path"), "profile_" + str(count)
        ),
    )


def generate_simulation_config(config: dict):
    if config.get("simulation_name") is None or config["sim_parameters"].get("match"):
        config = match_simulation(config=config)
    config["sim_parameters"]["pixel_size"] = config.get("pixel_size")
    return config


def match_simulation(config: dict):
    sim_config = deepcopy(config["sim_parameters"])
    # TODO: check for spherical here?
    sim_config["sim_height"] = config.get("length")
    sim_config["sim_width"] = config.get("diameter")

    config["sim_parameters"] = sim_config

    return config


def save_config(config: dict, filename: str):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_profile(profile: dict, filename: str):
    np.save(filename, profile)


def load_system_config(config: dict, system_config_path: str = None):

    if system_config_path is None:
        raise ValueError("System config path must be provided.")

    system_config = j_utils.load_yaml_config(config_filename=system_config_path)

    if sys.platform == "win32":
        base_path = system_config.get("windows")
    elif sys.platform == "linux":
        base_path = system_config.get("linux")
    else:
        raise ValueError("Unknown operating system.")

    system_config["base_path"] = base_path

    system_config["configs_path"] = os.path.join(
        system_config.get("base_path"),
        system_config.get("juno_custom_path"),
        "simulations",
        config.get("directory_name"),
        "configs",
    )
    return system_config
