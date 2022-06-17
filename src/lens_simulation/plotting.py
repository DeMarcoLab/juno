import os
import glob

import matplotlib.pyplot as plt
import numpy as np

import imageio

from PIL import Image
from pathlib import Path
from lens_simulation import utils
from lens_simulation.Lens import Lens
from lens_simulation.structures import (
    SimulationResult,
    SimulationStage,
    SimulationParameters,
)

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

    # save top-down
    fig = utils.plot_simulation(
        arr=result.top_down,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.start_distance,
        finish_distance=stage.finish_distance,
    )

    utils.save_figure(fig, os.path.join(save_path, "topdown.png"))
    plt.close(fig)

    fig = utils.plot_simulation(
        np.log(result.top_down + 10e-12),
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.start_distance,
        finish_distance=stage.finish_distance,
    )

    utils.save_figure(fig, os.path.join(save_path, "log_topdown.png"))
    plt.close(fig)

    fig = utils.plot_simulation(
        arr=result.side_on,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.start_distance,
        finish_distance=stage.finish_distance,
    )
    utils.save_figure(fig, os.path.join(save_path, "sideon.png"))
    plt.close(fig)

    if result.freq_arr is not None:
        fig = utils.plot_image(
            result.freq_arr,
            "Frequency Array",
            save=True,
            fname=os.path.join(save_path, "freq.png"),
        )
        plt.close(fig)

    if result.delta is not None:
        fig = utils.plot_image(
            result.delta,
            "Delta Profile",
            save=True,
            fname=os.path.join(save_path, "delta.png"),
        )
        plt.close(fig)

    if result.phase is not None:
        utils.plot_image(
            result.phase,
            "Phase Profile",
            save=True,
            fname=os.path.join(save_path, "phase.png"),
        )
        plt.close(fig)

    if result.lens is not None:
        fig = utils.plot_lens_profile_2D(result.lens)
        utils.save_figure(fig, fname=os.path.join(save_path, "lens_profile.png"))

        fig = utils.plot_lens_profile_slices(result.lens)
        utils.save_figure(fig, fname=os.path.join(save_path, "lens_slices.png"))

    # save propagation gifs
    try:
        utils.save_propagation_gif(save_path)
        utils.save_propagation_slices_gif(save_path)
    except:
        pass


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

        sim[i, :, :] = slice

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
