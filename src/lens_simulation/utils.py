import matplotlib.pyplot as plt
import numpy as np

import logging
import datetime
import time

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
        arr (np.ndarray): the simulation output arrays [n_slices, height, width]
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

    if lens.aperture is not None:
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
    if tickstyle is not "plain":
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



def plot_lens_profile_slices(lens: Lens, max_height: float = None, title: str = "Lens Profile Slices", facecolor: str = "#ffffff") -> plt.Figure:
    # TODO: add proper distances to plot
    """Plot slices of a two-dimensional lens at one-eighth, one-quarter and one-half distances"""

    if isinstance(lens, np.ndarray):
        lens_profile = lens
    if isinstance(lens, Lens):
        lens_profile = lens.profile
    else:
        raise TypeError("Non-Lens passed")

    thirty_two_px = lens.profile.shape[0] // 32
    sixteen_px = lens.profile.shape[0] // 16
    sixth_px = lens_profile.shape[0] // 8
    quarter_px = lens_profile.shape[0] // 4
    mid_px = lens_profile.shape[0] // 2

    # TODO: slice in the other directions

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


def save_propagation_gif(path: str):
    """Save a gif of the propagation"""


    search_path = os.path.join(path, "*mm.npy")

    filenames = sorted(glob.glob(search_path))
    images = []
    for fname in filenames:

        slice = np.load(fname)
        img = Image.fromarray(slice)

        images.append(img)

    save_path = os.path.join(path, "propagation.gif")
    imageio.mimsave(save_path, images, duration=0.2)

def save_propagation_slices_gif(path: str) -> None:
    """Save vertical and horizontal simulation slices as gif"""
    filenames = sorted(glob.glob(os.path.join(path, "*mm.npy")))

    # TODO: might not be possible for very large sims to load full sim,
    # will need to come up with another way to load slices in right format
    sim = None
    for i, fname in enumerate(filenames):

        slice = np.load(fname)

        if sim is None:
            sim = np.zeros(shape=(len(filenames), *slice.shape), dtype=np.float32)

        sim[i,:, :] = slice

    # normalise sim values
    sim = (sim - np.mean(sim)) / np.std(sim)
    # sim = (sim - np.min(sim)) / (np.max(sim) - np.min(sim))
    # sim = np.clip(sim, 0.5, np.max(sim))
    # sim = sim.astype(np.uint16)

    # save horizontal slices
    horizontal = []
    for i in range(sim.shape[2]):

        slice = sim[:, :, i]
        horizontal.append(slice)

    save_path = os.path.join(path, "horizontal.gif")
    imageio.mimsave(save_path, horizontal, duration=0.05)

    # save vertical slices
    vertical = []
    for i in range(sim.shape[1]):

        slice = sim[:, i, :]
        vertical.append(slice)

    save_path = os.path.join(path, "vertical.gif")
    imageio.mimsave(save_path, vertical, duration=0.05)

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
    # TODO move to validation,
    # TODO: medium, stages, see _format_dictionary
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


def _calculate_num_of_pixels(width: float, pixel_size: float, odd: bool = True) -> int:
    """Calculate the number of pixels for a given width and pixel size

    Args:
        width (float): the width of the image (metres)
        pixel_size (float): the size of the pixels (metres)
        odd (bool, optional): force the n_pixels to be odd. Defaults to True.

    Returns:
        int: the number of pixels in the image distance
    """
    n_pixels = int(width / pixel_size)

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


def plot_apeture_masks(lens: Lens) -> plt.Figure:

    fig, ax = plt.subplots(2, 3, figsize=(10, 7.5))

    plt.suptitle("Lens Aperture Masks")
    ax[0, 0].imshow(lens.non_lens_mask, cmap="plasma")
    ax[0, 0].set_title("non_lens_area")

    ax[0, 1].imshow(lens.truncation_aperture_mask, cmap="plasma")
    ax[0, 1].set_title("truncation_aperture")

    ax[1, 0].imshow(lens.custom_aperture_mask, cmap="plasma")
    ax[1, 0].set_title("custom_aperture")

    ax[1, 1].imshow(lens.sim_aperture_mask, cmap="plasma")
    ax[1, 1].set_title("sim_aperture")


    ax[0, 2].imshow(lens.aperture, cmap="plasma")
    ax[0, 2].set_title("full_aperture")

    # lens.profile[lens.aperture] = 0
    # lens.profile[lens.non_lens_mask.astype(bool)] = 1
    # lens.profile[lens.truncation_aperture_mask.astype(bool)] = 2
    # lens.profile[lens.custom_aperture_mask.astype(bool)] = 3
    # lens.profile[lens.sim_aperture_mask.astype(bool)] = 4
    ax[1, 2].imshow(lens.profile, cmap="plasma")
    ax[1, 2].set_title("lens_profile")

    return fig

def plot_lens_modifications(lens):
    from lens_simulation.Lens import check_modification_masks

    lens = check_modification_masks(lens)

    fig, ax = plt.subplots(2, 2, figsize=(7.5, 7.5))

    plt.suptitle("Lens Modifcations")
    ax[0, 0].imshow(lens.grating_mask, cmap="plasma")
    ax[0, 0].set_title("grating mask")

    ax[0, 1].imshow(lens.escape_mask, cmap="plasma")
    ax[0, 1].set_title("escape mask")

    ax[1, 0].imshow(lens.truncation_mask, cmap="plasma")
    ax[1, 0].set_title("truncation mask")

    ax[1, 1].imshow(lens.profile, cmap="plasma")
    ax[1, 1].set_title("lens profile")

    return fig

