import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import os 
import json
import yaml

from lens_simulation.Lens import Lens

# TODO: visualisation
# visualisation between lens and sim data is inconsistent, 
# light comes from bottom for lens profile, and top for sim result.
# need to make it more consistent, and file a way to tile the results into the sim setup for visualisation

# initial beam -> lens -> sim -> lens -> sim


def plot_simulation(
    arr: np.ndarray,
    width: int,
    height: int,
    pixel_size_x: float,
    start_distance: float,
    finish_distance: float,
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
    plt.title(f"Simulation Output ({width}x{height})")
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


def save_figure(fig, fname: str = "img.png") -> None:
    # TODO: clean up the implementation (no reference to fig...)
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # plt.savefig(fname)
    fig.savefig(fname)


def plot_interactive_simulation(arr: np.ndarray):
    # TODO: make croppable?
    
    fig = px.imshow(arr)
    return fig

def plot_lenses(lens_dict: dict) -> None:
    # plot lens profiles
    for name, lens in lens_dict.items():
        # fig, ax = plt.Figure()
        plt.title("Lens Profiles")
        plt.plot(lens.profile, label=name)
        plt.legend(loc="best")
        plt.plot()


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

def load_config(config_filename):
    with open(config_filename, "r") as f:
        conf = yaml.full_load(f)


    # validation
    # TODO: medium, stages, see _format_dictionary
    # convert all height and exponent values to float
    for i, lens in enumerate(conf["lenses"]):
        for param in ["height", "exponent"]:
            if isinstance(lens[param], list):
                for j, h in enumerate(lens[param]):
                    conf["lenses"][i][param][j] = float(h)

    return conf


def plot_lens_profile_2D(lens: Lens):
    # TODO: add proper distances to plot
    fig = plt.figure()
    plt.title("Lens Profile (Two-Dimensional)")
    plt.imshow(lens.profile, cmap="plasma")
    plt.colorbar()
    
    return fig


def plot_lens_profile_slices(lens: Lens) -> plt.Figure:
    # TODO: add proper distances to plot
    """Plot slices of a two-dimensional lens at one-sixth, one-quarter and one-half distances"""
    lens_profile = lens.profile
    sixth_px = lens_profile.shape[0] // 6
    quarter_px = lens_profile.shape[0] // 4
    mid_px = lens_profile.shape[0] // 2

    fig = plt.figure()
    plt.title("Lens Profile Slices")
    plt.plot(lens_profile[sixth_px, :], "r--", label="Sixth") 
    plt.plot(lens_profile[quarter_px, :], "g--", label="Quarter")
    plt.plot(lens_profile[mid_px, :], "b--", label="MidPoint")
    plt.legend(loc="best")
    
    return fig