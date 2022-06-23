import glob
import logging
import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image, ImageDraw

from lens_simulation import utils
from lens_simulation.Lens import Lens
from lens_simulation.structures import (SimulationParameters, SimulationResult,
                                        SimulationStage)

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



def plot_lens_profile_slices(lens: Lens, max_height: float = None, title: str = "Lens Profile Slices", facecolor: str = "#ffffff", dim: int = 0) -> plt.Figure:
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
    
def save_result_plots(
    result: SimulationResult,
    stage: SimulationStage,
    parameters: SimulationParameters,
    save_path: Path,
):
    """Plot and save the simulation results

    Args:
        result (SimulationResult): _description_
        stage (SimulationStage): _description_
        parameters (SimulationParameters): _description_
        save_path (Path): _description_
    """

    sim = result.sim
    top_down, side_on = create_sim_views(sim)

    # save top-down
    fig = plot_simulation(
        arr=top_down,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.distances[0],
        finish_distance=stage.distances[-1],
    )

    save_figure(fig, os.path.join(save_path, "topdown.png"))
    plt.close(fig)

    fig = plot_simulation(
        np.log(top_down + 10e-12),
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.distances[0],
        finish_distance=stage.distances[-1],
    )

    save_figure(fig, os.path.join(save_path, "log_topdown.png"))
    plt.close(fig)

    fig = plot_simulation(
        arr=side_on,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.distances[0],
        finish_distance=stage.distances[-1],
    )
    save_figure(fig, os.path.join(save_path, "sideon.png"))
    plt.close(fig)

    if result.freq_arr is not None:
        fig = plot_image(
            result.freq_arr,
            "Frequency Array",
            save=True,
            fname=os.path.join(save_path, "freq.png"),
        )
        plt.close(fig)

    if result.delta is not None:
        fig = plot_image(
            result.delta,
            "Delta Profile",
            save=True,
            fname=os.path.join(save_path, "delta.png"),
        )
        plt.close(fig)

    if result.phase is not None:
        plot_image(
            result.phase,
            "Phase Profile",
            save=True,
            fname=os.path.join(save_path, "phase.png"),
        )
        plt.close(fig)

    if result.lens is not None:
        fig = plot_lens_profile_2D(result.lens)
        save_figure(fig, fname=os.path.join(save_path, "lens_profile.png"))
        plt.close(fig)

        fig = plot_lens_profile_slices(result.lens)
        save_figure(fig, fname=os.path.join(save_path, "lens_slices.png"))
        plt.close(fig)
            
        fig = plot_apeture_masks(result.lens)
        save_figure(fig, fname=os.path.join(save_path, "lens_aperture.png"))
        plt.close(fig)
        
        try:
            fig = plot_lens_modifications(result.lens)
            save_figure(fig, fname=os.path.join(save_path, "lens_modifications.png"))
            plt.close(fig)
        except: 
            pass # cant plot apertures and truncation for beam...

    # save propagation gifs
    try:
        save_propagation_gif(os.path.join(save_path, "sim.zarr"))
    except Exception as e:
        logging.error(f"Error during plotting GIF: {e}")

def create_sim_views(sim: np.ndarray, px_h: int = None, px_v: int = None) -> tuple:
    """Create vertical and horizontal slices of the simulation"""
    if px_h is None:
        px_h = sim.shape[1] // 2
    if px_v is None:
        px_v = sim.shape[2] // 2

    # calculate views
    top_down = sim[:, px_h, :]
    side_on = sim[:, :, px_v]

    return top_down, side_on


def plot_apeture_masks(lens: Lens) -> plt.Figure:

    fig, ax = plt.subplots(2, 3, figsize=(10, 7.5))

    plt.suptitle("Lens Aperture Masks")
    ax[0, 0].imshow(lens.non_lens_mask, cmap="plasma")
    ax[0, 0].set_title("non_lens_area")

    ax[0, 1].imshow(lens.truncation_aperture_mask, cmap="plasma")
    ax[0, 1].set_title("truncation_aperture")

    ax[1, 0].imshow(lens.custom_aperture_mask, cmap="plasma")
    ax[1, 0].set_title("custom_aperture")

    ax[1, 1].imshow(lens.loaded_aperture, cmap="plasma")
    ax[1, 1].set_title("loaded_aperture")

    ax[0, 2].imshow(lens.sim_aperture_mask, cmap="plasma")
    ax[0, 2].set_title("sim_aperture")

    ax[1, 2].imshow(lens.aperture, cmap="plasma")
    ax[1, 2].set_title("full_aperture")

    return fig


def plot_lens_modifications(lens: Lens) -> plt.Figure:
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


def plot_simulation_setup(config: dict) -> plt.Figure:

    arr = None
    sim_height = config["sim_parameters"]["sim_height"]
    pixel_size = config["sim_parameters"]["pixel_size"]
    sim_n_pixels_h = utils._calculate_num_of_pixels(sim_height, pixel_size, True)

    for conf in config["stages"]:

        # get stage info
        output_medium = conf["output"]
        sd = conf["start_distance"]
        fd = conf["finish_distance"]
        total = fd - sd
        n_pixels = utils._calculate_num_of_pixels(total, pixel_size, True)

        # get lens info
        lens_name = conf["lens"]
        for lc in config["lenses"]:
            if lens_name == lc["name"]:
                lens_height = lc["height"]
                lens_medium = lc["medium"]
                break

        lens_n_pixels_z = utils._calculate_num_of_pixels(lens_height, pixel_size, True)

        # create arr
        output = np.ones(shape=(sim_n_pixels_h, n_pixels)) * output_medium
        lens = np.ones(shape=(sim_n_pixels_h, lens_n_pixels_z)) * lens_medium

        if arr is None:
            arr = arr = np.hstack([lens, output])
        else:
            arr = np.hstack([arr, lens, output])

    # create plot
    fig = plt.figure(figsize=(15, 2))
    plt.imshow(arr, cmap="plasma")
    clb = plt.colorbar()
    clb.ax.set_title("Medium")
    plt.title("Simulation Stages")
    plt.xlabel("Propagation Distance (px)")
    plt.ylabel("Simulation Height (px)")
    # TODO: correct the propagation distances to mm
    return fig

def save_propagation_gif(path: str, vert: bool = False, hor: bool = False):
    """Save a gif of the propagation"""

    sim = utils.load_simulation(path)
    # ref: https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
    # TODO: fix this properly: not converting to np.uint8 correctly, scaling is off
    # normalise
    # sim = (sim - np.mean(sim)) / (np.max(sim) - np.min(sim)) * np.iinfo(np.uint8).max

    if sim.dtype == np.float16:    
        sim = sim.astype(np.uint8)

    save_path = os.path.join(os.path.dirname(path), "propagation.gif")
    imageio.mimsave(save_path, sim, duration=0.2)
    
    if vert:
        vertical_save_path = os.path.join(os.path.dirname(path), "vertical.gif")
        imageio.mimsave(vertical_save_path, np.swapaxes(sim, 0, 1), duration=0.2)
    if hor:
        horizontal_save_path = os.path.join(os.path.dirname(path), "horizontal.gif")
        imageio.mimsave(horizontal_save_path, np.swapaxes(sim, 0, 2), duration=0.2)
