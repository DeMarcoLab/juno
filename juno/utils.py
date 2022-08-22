import numpy as np

import logging
import datetime
import time

import pandas as pd
import os
import json
import yaml
import petname

import zarr
from juno import validation

from pathlib import Path


#################### DATA / IO ####################
def load_simulation(path):
    """Load a simulation using zarr"""
    sim = zarr.open(path, mode="r")

    return sim

def load_np_arr(fname: str) -> np.ndarray:
    """Load a numpy array from disk"""
    arr = np.load(fname)
    return arr

def save_metadata(config: dict, log_dir: str) -> None:
    
    # serialisable 
    config["sim_id"] = str(config.get("sim_id"))
    config["run_id"] = str(config.get("run_id"))

    # save as json
    with open(os.path.join(log_dir, "metadata.json"), "w") as f:
        json.dump(config, f, indent=4)

def load_metadata(path: str):
    metadata_fname = os.path.join(path, "metadata.json")

    with open(metadata_fname, "r") as f:
        metadata = json.load(f)

    return metadata


def save_simulation(sim: np.ndarray, fname: Path) -> None:
    """Save the simulation array as a numpy array

    Args:
        sim (_type_): _description_
        fname (_type_): _description_
    """
    # TODO: use npz (compressed)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    np.save(fname, sim)


def load_yaml_config(config_filename) -> dict:
    with open(config_filename, "r") as f:
        config = yaml.full_load(f)

    return config

def load_config(config_filename: str, validate: bool = True):

    config = load_yaml_config(config_filename)

    # validation
    if validate:
        config = validation._validate_simulation_config(config)

    return config



def load_config_struct(config_filename: str):
    """Load the config as a struct"""
    # from juno.structures import SimulationConfig
    
    config = load_config(config_filename)

    config_stuct = config

    return config_stuct




def load_simulation_config(config_filename: str = "config.yaml") -> dict:
    """Load the default configuration ready to simulate.

    Args:
        config_filename (str, optional): config filename. Defaults to "config.yaml".

    Returns:
        dict: configuration as dictionary formatted for simulation
    """

    conf = load_config(config_filename)

    run_id = petname.generate(3)  # run_id is for when running a batch of sims, each sim has unique id
    data_path = os.path.join(conf["options"]["log_dir"],  str(run_id))
    config = {"run_id": run_id,
                "parameters": None,
                "log_dir": data_path,
                "sim_parameters": conf["sim_parameters"],
                "options": conf["options"],
                "beam": conf["beam"],
                "mediums": conf["mediums"],
                "lenses": conf["lenses"],
                "stages": conf["stages"]}

    return config

######################## DATAFRAME ########################

def load_simulation_data(path):
    """Load all simulation metadata into a single dataframe"""
    metadata = load_metadata(path)

    # QUERY: add prefix for lens_ and stage_ ? might need to adjust config

    # individual stage metadata
    df_stages = pd.DataFrame.from_dict(metadata["stages"])
    df_stages["stage"] = df_stages.index + 1

    df_lens = pd.DataFrame.from_dict(metadata["lenses"])
    df_lens = df_lens.rename(columns={"name": "lens"})

    # lens modifications

    # gratings
    grats = []
    for grat in list(df_lens["grating"]):
        if grat is None:
            grat = {"width": None, "distance": None, "depth": None, "x": None, "y": None, "centred": None}

        grats.append(grat)

    df_grat = pd.DataFrame.from_dict(grats)
    df_grat = df_grat.add_prefix("grating_")
    df_lens = pd.concat([df_lens, df_grat], axis=1)

    # truncation
    truncs = []
    for trunc in list(df_lens["truncation"]):
        if trunc is None:
            trunc = {"height": None, "radius": None, "type": None, "aperture": None}

        truncs.append(trunc)

    df_trunc = pd.DataFrame.from_dict(truncs)
    df_trunc = df_trunc.add_prefix("truncation_")
    df_lens = pd.concat([df_lens, df_trunc], axis=1)

    # aperture
    apertures = []
    for aperture in list(df_lens["aperture"]):
        if aperture is None:
            aperture = {"inner": None, "outer": None, "type": None, "invert": None}

        apertures.append(aperture)

    df_aperture = pd.DataFrame.from_dict(apertures)
    df_aperture = df_aperture.add_prefix("aperture_")
    df_lens = pd.concat([df_lens, df_aperture], axis=1)

    # common metadata
    df_beam = pd.DataFrame.from_dict([metadata["beam"]])
    df_beam = df_beam.add_prefix("beam_")
    df_parameters = pd.DataFrame.from_dict([metadata["sim_parameters"]])
    df_options = pd.DataFrame.from_dict([metadata["options"]])
    df_common = pd.concat([df_beam, df_parameters, df_options], axis=1)
    df_common["petname"] = metadata["petname"]


    # join dataframes
    df_join = pd.merge(df_stages, df_lens, on="lens")
    df_join["petname"] = metadata["petname"]
    df_join = pd.merge(df_join, df_common, on="petname")


    # common parameters
    df_join["sim_id"] = metadata["sim_id"]
    df_join["run_id"] = metadata["run_id"]
    df_join["run_petname"] = metadata["run_petname"]
    df_join["log_dir"] = metadata["log_dir"]
    df_join["path"] = os.path.join(metadata["log_dir"], metadata["petname"])
    df_join["started"] = metadata["started"]
    df_join["finished"] = metadata["finished"]


    df_join["lens"] = df_join["lens"].astype(str)

    return df_join


def load_run_simulation_data(directory):
    """Join all simulations metadata into a single dataframe

    Args:
        directory (_type_): _description_
    """
    sim_directories = [os.path.join(directory, path) for path in os.listdir(directory) if os.path.isdir(os.path.join(directory, path))]

    df = pd.DataFrame()

    for path in sim_directories:

        df_join = load_simulation_data(path)

        df = pd.concat([df, df_join],ignore_index=True).reset_index()
        df = df.drop(columns=["index"])

    # remove sweep columns from frame
    sweep_cols = [col for col in df.columns if "_stop" in col or "_step" in col]
    df = df.drop(columns=sweep_cols)

    return df

def save_dataframe(directory: Path, fname: str = "data.csv"):
    """Save all the simulation data from dataframe to csv."""
    df: pd.DataFrame = load_run_simulation_data(directory)

    df.to_csv(os.path.join(directory, fname))


def load_dataframe(path: Path):
    """Load dataframe from disk"""
    df = pd.read_csv(path)

    return df

################ UTILITIES

def _calculate_num_of_pixels(width: float, pixel_size: float, odd: bool = True) -> int:
    """Calculate the number of pixels for a given width and pixel size

    Args:
        width (float): the width of the image (metres)
        pixel_size (float): the size of the pixels (metres)
        odd (bool, optional): force the n_pixels to be odd. Defaults to True.

    Returns:
        int: the number of pixels in the image distance
    """
    n_pixels = round(width / pixel_size) # NOTE: rounding the width first, to prevent flooring.

    # n_pixels must be odd (symmetry).
    if odd and n_pixels % 2 == 0:
        n_pixels += 1

    return n_pixels


def create_distance_map_px(w: int, h: int) -> np.ndarray:
    x = np.arange(0, w)
    y = np.arange(0, h)

    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(((w / 2) - X) ** 2 + ((h / 2) - Y) ** 2)

    return distance



def current_timestamp() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d.%H%M%S')

# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
def configure_logging(save_path='', log_filename='logfile', log_level=logging.INFO):
    """Log to the terminal and to file simultaneously."""
    timestamp = current_timestamp()

    logfile = os.path.join(save_path, f"{log_filename}.log")

    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(),
        ])

    return logfile

def pad_to_equal_size(small: np.ndarray, large: np.ndarray, value: int = 0) -> tuple:
    """Determine the amount to pad to match size"""
    sh, sw = small.shape
    lh, lw = large.shape
    ph, pw = int((lh - sh) // 2), int((lw - sw) // 2)

    padded = np.pad(small, pad_width=((ph, ph), (pw, pw)), mode="constant", constant_values=value)

    return padded

