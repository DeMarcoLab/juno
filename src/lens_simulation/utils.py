import matplotlib.pyplot as plt
import numpy as np

import logging
import datetime
import time

import pandas as pd
import plotly.express as px
import os
import json
import yaml
import petname

import glob
import imageio
from PIL import Image

from lens_simulation.Lens import Lens
from lens_simulation import validation

from pathlib import Path

# TODO: visualisation
# visualisation between lens and sim data is inconsistent,
# light comes from bottom for lens profile, and top for sim result.
# need to make it more consistent, and file a way to tile the results into the sim setup for visualisation

# initial beam -> lens -> sim -> lens -> sim

#################### PLOTTING ####################

def plot_simulation(
    arr: np.ndarray,
    width: int=None,
    height: int=None,
    pixel_size_x: float=1e-6,
    start_distance: float=0e-6,
    finish_distance: float=10e-6,
) -> plt.Figure:
    """Plot the output simulation array from the top down perspective.

    Args:
        arr (np.ndarray): the simulation output arrays [n_steps, height, width]
        width (int): [the horizontal distance to plot]
        height (int): [the depth of the simulation to plot]
        pixel_size_x (float): [simulation pixel size]
        start_distance (float): [the start distance for the simulation propagation]
        finish_distance (float): [the finish distance for the simulation propagation]

    Returns:
        [Figure]: [matplotlib Figure of the simulation plot]
    """
    if width is None:
        width = arr.shape[1]
    if height is None:
        height = arr.shape[0]

    arr_resized, min_h, max_h = crop_image(arr, width, height)

    # calculate extents (xlabel, ylabel)
    min_x = -arr_resized.shape[1] / 2 * pixel_size_x / 1e-6
    max_x = arr_resized.shape[1] / 2 * pixel_size_x / 1e-6

    # nb: these are reversed because the light comes from top...
    dist = finish_distance - start_distance

    min_h_frac = min_h / arr.shape[0]
    max_h_frac = max_h / arr.shape[0]

    min_y = (start_distance + max_h_frac * dist) / 1e-3
    max_y = (start_distance + min_h_frac * dist) / 1e-3

    fig = plt.figure()
    plt.imshow(
        arr_resized,
        extent=[min_x, max_x, min_y, max_y],
        interpolation="spline36",
        aspect="auto",
        cmap="jet",
    )
    plt.title(f"Simulation Output ({height}x{width})")
    plt.ylabel("Distance (mm)")
    plt.xlabel("Distance (um)")
    plt.colorbar()

    return fig

def crop_image(arr, width, height):
    """Crop the simulation image to the required dimensions."""

    if arr.ndim == 3:
        vertical_index = arr.shape[1] // 2 # midpoint (default)
        arr = arr[:, vertical_index, :] # horizontal plane slice

    min_h, max_h = arr.shape[0] // 2 - height // 2, arr.shape[0] // 2 + height // 2
    min_w, max_w = arr.shape[1] // 2 - width // 2, arr.shape[1] // 2 + width // 2

    arr_resized = arr[min_h:max_h, min_w:max_w]
    return arr_resized, min_h,max_h


def plot_image(arr: np.ndarray, title: str = "Image Title", save: bool = False, fname: str = None) -> plt.Figure:
    """Plot an image and optionally save."""
    fig = plt.figure()
    plt.imshow(arr)
    plt.title(title)
    plt.colorbar()
    if save:
        save_figure(fig, fname)

    return fig


def save_figure(fig, fname: str = "img.png") -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname)


def plot_interactive_simulation(arr: np.ndarray):

    fig = px.imshow(arr)
    return fig


def plot_lens_profile_2D(lens: Lens, title="", facecolor="#ffffff", tickstyle="sci",
                         cmap=None, extent=None, colorbar_ticks=None):
    """Plots a lens profile using plot_array_2D, including aperture hatching"""
    # TODO: Add tests for this
    if not isinstance(lens, Lens):
        raise TypeError('plot_lens_profile_2D requires a Lens object')

    fig = plot_array_2D(array=lens.profile, title=title, facecolor=facecolor,
    tickstyle=tickstyle, cmap=cmap, extent=extent, colorbar_ticks=colorbar_ticks)

    axes = fig.axes[0]

    # weird bug with (1, 1) shape as array is false
    if lens.aperture is not None and lens.aperture.shape[0] != 1 and lens.aperture.shape[1] != 1:
        aperture = np.ma.array(lens.aperture, mask=[lens.aperture == 0])
        axes.contourf(aperture, hatches=["x"], extent=extent, cmap="gray")

    return fig


def plot_array_2D(array: np.ndarray, title="", facecolor="#ffffff", tickstyle="sci", cmap=None, extent=None, colorbar_ticks=None):
    """Plots an ndarray"""
    if not isinstance(array, np.ndarray):
        raise TypeError('plot_array_2D requires a numpy.ndarray object')

    fig = plt.figure()
    fig.set_facecolor(facecolor)

    # set up axes
    gridspec = fig.add_gridspec(1, 1)
    axes = fig.add_subplot(gridspec[0], title=title)
    axes.ticklabel_format(axis="both", style=tickstyle, scilimits=(0, 0), useMathText=True)
    axes.locator_params(nbins=4)

    # reposition scientific notation on x axis
    if tickstyle != "plain":
        x_ext = axes.xaxis.get_offset_text()
        x_ext.set_x(1.2)

    # set up colorbar positioning/size
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes)
    cax=divider.append_axes('right', size='5%', pad=0.1)

    # plot image
    image = axes.imshow(array, aspect='auto', extent=extent, cmap=cmap)

    # plot colorbar
    fig.colorbar(image, cax=cax, orientation='vertical', ticks=colorbar_ticks)

    return fig



def plot_lens_profile_steps(lens: Lens, max_height: float = None, title: str = "Lens Profile Slices", facecolor: str = "#ffffff", dim: int = 0) -> plt.Figure:
    # TODO: add proper distances to plot
    """Plot slices of a two-dimensional lens at one-eighth, one-quarter and one-half distances"""

    if isinstance(lens, np.ndarray):
        lens_profile = lens
    if isinstance(lens, Lens):
        lens_profile = lens.profile
    else:
        raise TypeError("Non-Lens passed")

    thirty_two_px = lens.profile.shape[dim] // 32
    sixteen_px = lens.profile.shape[dim] // 16
    sixth_px = lens_profile.shape[dim] // 8
    quarter_px = lens_profile.shape[dim] // 4
    mid_px = lens_profile.shape[dim] // 2

    fig = plt.figure()
    fig.set_facecolor(facecolor)
    plt.title(title)
    plt.plot(lens_profile[mid_px, :], "b--", label="0.5")
    plt.plot(lens_profile[quarter_px, :], "g--", label="0.25")
    plt.plot(lens_profile[sixth_px, :], "r--", label="0.125")
    plt.plot(lens_profile[sixteen_px, :], "c--", label="0.0625")
    plt.plot(lens_profile[thirty_two_px, :], "m--", label="0.03125")
    plt.ylim([0, max_height])
    plt.legend(loc="best")

    return fig



#################### DATA ####################


def load_simulation(filename):
    sim = np.load(filename)
    return sim

def save_metadata(config: dict, log_dir: str) -> None:
    # serialisable
    if "sim_id" in config:
        config["sim_id"] = str(config["sim_id"])
    config["run_id"] = str(config["run_id"])

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

def load_config(config_filename):
    with open(config_filename, "r") as f:
        config = yaml.full_load(f)

    config = validation._validate_simulation_config(config)
    # validation

    # TODO move to validation, finish
    # convert all height and exponent values to float
    for i, lens in enumerate(config["lenses"]):
        for param in ["height", "exponent"]:
            if isinstance(lens[param], list):
                for j, h in enumerate(lens[param]):
                    config["lenses"][i][param][j] = float(h)

    return config

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

    # df = df.drop(columns=["level_0", "index", "options"])

    return df





################

def _calculate_num_of_pixels(width: float, pixel_size: float, odd: bool = True) -> int:
    """Calculate the number of pixels for a given width and pixel size

    Args:
        width (float): the width of the image (metres)
        pixel_size (float): the size of the pixels (metres)
        odd (bool, optional): force the n_pixels to be odd. Defaults to True.

    Returns:
        int: the number of pixels in the image distance
    """
    n_pixels = int(round(width / pixel_size))

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

