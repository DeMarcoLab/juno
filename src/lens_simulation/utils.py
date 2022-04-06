import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import os 
import json
import yaml


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
    """Plot the output simulation array.

    Args:
        arr (np.ndarray): the simulation output arrays
        width (int): [description]
        height (int): [description]
        pixel_size_x (float): [description]
        start_distance (float): [description]
        finish_distance (float): [description]

    Returns:
        [type]: [description]
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
    min_h, max_h = arr.shape[0] // 2 - height // 2, arr.shape[0] // 2 + height // 2
    min_w, max_w = arr.shape[1] // 2 - width // 2, arr.shape[1] // 2 + width // 2

    arr_resized = arr[min_h:max_h, min_w:max_w]
    return arr_resized, min_h,max_h


def save_figure(fig, fname: str = "img.png") -> None:

    os.makedirs(os.path.dirname(fname), exist_ok=True)

    plt.savefig(fname)


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
    # TODO: medium, stages
    # convert all height and exponent values to float
    for i, lens in enumerate(conf["lenses"]):
        for param in ["height", "exponent"]:
            if isinstance(lens[param], list):
                for j, h in enumerate(lens[param]):
                    conf["lenses"][i][param][j] = float(h)

    return conf