import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import os 
import json
import yaml
import petname

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

def plot_lenses(simulation_lenses: dict) -> None:
    # plot lens profiles
    for name, lens in simulation_lenses.items():
        # fig, ax = plt.Figure()
        plt.title("Lens Profiles")
        plt.plot(lens.profile, label=name)
        plt.legend(loc="best")
        plt.plot()

def plot_lens_profile_2D(lens: Lens):
    # TODO: add proper distances to plot

    if isinstance(lens, np.ndarray):
        lens_profile = lens
    if isinstance(lens, Lens):
        lens_profile = lens.profile

    fig = plt.figure()
    plt.title("Lens Profile (Two-Dimensional)")
    plt.imshow(lens_profile, cmap="plasma")
    plt.colorbar()
    
    return fig


def plot_lens_profile_slices(lens: Lens, max_height: float = None) -> plt.Figure:
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
    plt.title("Lens Profile Slices")
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