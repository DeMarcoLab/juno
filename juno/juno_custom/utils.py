import os

import numpy as np
import yaml


def save_configuration(config: dict, profile: np.ndarray = None):
    save_base = os.path.join(config["folder"], config["name"])

    os.makedirs(config["folder"], exist_ok=True)

    with open(save_base + ".yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    if profile is not None:
        np.save(save_base + ".npy", profile)


def load_configuration(config_filename: str):
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    return config
