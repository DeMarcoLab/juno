import glob
import logging
import os
from pathlib import Path
from pprint import pprint

import dask.array as da
import imageio
import matplotlib.pyplot as plt
import napari
import numpy as np
import zarr

from juno import utils
from juno.beam import Beam, generate_beam
from juno.Lens import Lens, generate_lens
from juno.Medium import Medium
from juno.structures import (SimulationParameters, SimulationResult,
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
        cmap="turbo",
    )
    plt.title(f"Simulation Output ({height}x{width})")
    plt.ylabel("Distance (mm)")
    plt.xlabel("Distance (um)")
    plt.colorbar()

    return fig


def cross_section_image(arr: np.ndarray, axis: int = 0, prop: float = 0.5) -> np.ndarray:

    if arr.ndim != 2:
        raise ValueError(f"Only two-dimensional arrays are supported. The current array is {arr.ndim} dimensions.")

    if axis not in [0, 1]:
        raise ValueError(f"Only axis [0, 1] are supported. The axis {axis} is not supported.")

    if prop < 0 or prop > 1.0:
        raise ValueError(f"Proporation must be between 0 - 1.0. The proporation was {prop}. ")

    # get centre pixel
    centre_px = arr.shape[axis] // 2

    if axis == 0:
        cross_section = arr[centre_px, :]
    if axis == 1:
        cross_section = arr[:, centre_px]

    return cross_section


def threshold_image(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold the image above the proporation of the maximum value"""
    threshold_value = threshold * np.max(arr)

    mask = arr > threshold_value
    image_threshold = arr * mask

    return image_threshold


def max_intensity_projection(sim: np.ndarray, axis: int = 0) -> np.ndarray:
    """Maximum intensity projection of the simulation along the provided axis."""
    if sim.ndim != 3:
        raise ValueError(f"Maximum intesnity projection only works for 3D arrays. This array is {sim.ndim} dimensions.")

    if axis not in [0, 1, 2]:
        raise ValueError(f"Only axis [0, 1, 2] are supported. The axis {axis} is not supported.")

    max_projection = np.max(sim, axis=axis)

    return max_projection



def slice_simulation_view(sim: np.ndarray, axis: int = 0, prop: float = 0.5) -> np.ndarray:
    """Slice the simulation into a view along the provided axis."""
    if sim.ndim != 3:
        raise ValueError(f"Slice simulation view only works for 3D arrays. This array is {sim.ndim} dimensions.")

    if axis not in [0, 1, 2]:
        raise ValueError(f"Only axis [0, 1, 2] are supported. The axis {axis} is not supported.")

    if prop < 0 or prop > 1.0:
        raise ValueError(f"Proporation must be between 0 - 1.0. The proporation was {prop}. ")

    px = int(prop * sim.shape[axis]) - 1

    # got to be a better way...
    if axis == 0:
        sim_view = sim[px, :, :]

    if axis == 1:
        sim_view = sim[:, px, :]

    if axis == 2:
        sim_view = sim[:, :, px]

    return sim_view



def crop_image_v3(arr: np.ndarray, width: float = 0.5, height: float = 0.5, x: float = 0.5, y: float = 0.5):
    """Crop the image proportionally based on the shape."""

    if arr.ndim != 2:
        raise ValueError(f"Crop image v2 only supported 2D arrays. This array is {arr.ndim} dimensions.")

    # default to centre of image
    x_px = int(arr.shape[1] * x)
    y_px = int(arr.shape[0] * y)
    w_px = int(arr.shape[1] * width)
    h_px = int(arr.shape[0] * height)

    # make sure clip ends up within bounds (must be difference of 1)
    min_h = np.clip(int(y_px - h_px // 2), 0, arr.shape[0] - 1) 
    max_h = np.clip(int(y_px + h_px // 2), 1, arr.shape[0])
    min_w = np.clip(int(x_px - w_px // 2), 0, arr.shape[1] - 1)
    max_w = np.clip(int(x_px + w_px // 2), 1, arr.shape[1])

    arr_resized = arr[min_h:max_h, min_w:max_w]

    return arr_resized, (min_h, max_h, min_w, max_w)

def crop_image_v2(arr: np.ndarray, width: int = None, height: int = None, x: int = None, y: int = None):
    """Crop the simulation image to the required dimensions."""

    if arr.ndim != 2:
        raise ValueError(f"Crop image v2 only supported 2D arrays. This array is {arr.ndim} dimensions.")

    # default to centre of image
    if x is None:
        x = arr.shape[1] // 2
    if y is None:
        y = arr.shape[0] // 2

    # make sure clip ends up within bounds
    min_h = np.clip(y - height // 2, 0, arr.shape[0]) 
    max_h = np.clip(y + height // 2, 0, arr.shape[0])
    min_w = np.clip(x - width // 2, 0, arr.shape[1])
    max_w = np.clip(x + width // 2, 0, arr.shape[1])

    arr_resized = arr[min_h:max_h, min_w:max_w]

    return arr_resized, (min_h, max_h, min_w, max_w)


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
    plt.imshow(arr, cmap="plasma", aspect="auto")
    plt.title(title)
    plt.colorbar()
    if save:
        save_figure(fig, fname)

    return fig


def save_figure(fig, fname: str = "img.png") -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname)



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
    top_down = slice_simulation_view(sim, axis=1, prop=0.5)
    side_on = slice_simulation_view(sim, axis=2, prop=0.5)

    # save top-down
    fig = plot_simulation(
        arr=top_down.T,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.distances[0],
        finish_distance=stage.distances[-1],
    )

    save_figure(fig, os.path.join(save_path, "topdown.png"))
    plt.close(fig)

    fig = plot_simulation(
        np.log(top_down + 10e-12).T,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.distances[0],
        finish_distance=stage.distances[-1],
    )

    save_figure(fig, os.path.join(save_path, "log_topdown.png"))
    plt.close(fig)

    fig = plot_simulation(
        arr=side_on.T,
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
            
        try:
            fig = plot_apeture_masks(result.lens)
            save_figure(fig, fname=os.path.join(save_path, "lens_aperture.png"))
            plt.close(fig)
        
            fig = plot_lens_modifications(result.lens)
            save_figure(fig, fname=os.path.join(save_path, "lens_modifications.png"))
            plt.close(fig)
        except: 
            pass # cant plot apertures and truncation for beam...

    # save propagation gifs #TODO: remove :(
    # try:
    #     save_propagation_gif(os.path.join(save_path, "sim.zarr"))
    # except Exception as e:
    #     logging.error(f"Error during plotting GIF: {e}")

    plt.close(fig)
    plt.close()

def plot_apeture_masks(lens: Lens) -> plt.Figure:

    fig, ax = plt.subplots(2, 3, figsize=(10, 7.5))

    plt.suptitle("Lens Aperture Masks")
    ax[0, 0].imshow(lens.non_lens_mask, cmap="plasma", aspect="auto")
    ax[0, 0].set_title("non_lens_area")

    ax[0, 1].imshow(lens.truncation_aperture_mask, cmap="plasma", aspect="auto")
    ax[0, 1].set_title("truncation_aperture")

    ax[1, 0].imshow(lens.custom_aperture_mask, cmap="plasma", aspect="auto")
    ax[1, 0].set_title("custom_aperture")

    ax[1, 1].imshow(lens.loaded_aperture, cmap="plasma", aspect="auto")
    ax[1, 1].set_title("loaded_aperture")

    ax[0, 2].imshow(lens.sim_aperture_mask, cmap="plasma", aspect="auto")
    ax[0, 2].set_title("sim_aperture")

    ax[1, 2].imshow(lens.aperture, cmap="plasma", aspect="auto")
    ax[1, 2].set_title("full_aperture")

    return fig


def plot_lens_modifications(lens: Lens) -> plt.Figure:
    from juno.Lens import check_modification_masks

    lens = check_modification_masks(lens)

    fig, ax = plt.subplots(2, 2, figsize=(7.5, 7.5))

    plt.suptitle("Lens Modifcations")
    ax[0, 0].imshow(lens.grating_mask, cmap="plasma", aspect="auto")
    ax[0, 0].set_title("grating mask")

    ax[0, 1].imshow(lens.escape_mask, cmap="plasma", aspect="auto")
    ax[0, 1].set_title("escape mask")

    ax[1, 0].imshow(lens.truncation_mask, cmap="plasma", aspect="auto")
    ax[1, 0].set_title("truncation mask")

    ax[1, 1].imshow(lens.profile, cmap="plasma", aspect="auto")
    ax[1, 1].set_title("lens profile")

    return fig

def plot_simulation_setup_v2(config: dict, medium_only: bool = False) -> plt.Figure:
    from juno.Simulation import generate_simulation_parameters, pad_simulation
  
    arr = None
    sim_height = config["sim_parameters"]["sim_height"]
    pixel_size = config["sim_parameters"]["pixel_size"]
    sim_wavelength = config["sim_parameters"]["sim_wavelength"]
    sim_n_pixels_h = utils._calculate_num_of_pixels(sim_height, pixel_size, True)

    # TODO: add beam profile?    
    parameters = generate_simulation_parameters(config)
    beam: Beam = generate_beam(config["beam"], parameters)

    sd, fd = beam.start_distance, beam.finish_distance
    n_pixels_beam = utils._calculate_num_of_pixels((fd-sd), pixel_size, True)

    if medium_only:
        beam_medium = beam.output_medium.refractive_index
    else:
        beam_medium = 0

    beam_profile = da.stack([np.clip(beam.lens.profile, 0, beam.lens.medium.refractive_index)] * 1)
    beam_output = da.stack([np.clip(beam.lens.profile, 0, beam_medium)] * (n_pixels_beam - 1))

    arr = da.vstack([beam_profile, beam_output])

    for conf in config["stages"]:

        # get stage info
        output_medium = conf["output"]
        sd = conf["start_distance"]
        fd = conf["finish_distance"]
        total = fd - sd
        n_pixels_output = utils._calculate_num_of_pixels(total, pixel_size, True)

        # get lens info
        lens_name = conf["lens"]
        for lc in config["lenses"]:
            if lens_name == lc["name"]:
                lens_medium = lc["medium"]
                lens = generate_lens(lc,   Medium(lens_medium, sim_wavelength), pixel_size)
                break

        # create arr
        if not medium_only:
            output_medium = 0

        # need to pad to equal size as sim
        # pad the lens profile to be the same size as the simulation
        lens = pad_simulation(lens, parameters=parameters)

        # apply all aperture masks
        lens.apply_aperture_masks()
        profile = create_3d_lens(lens) * lens_medium

        # need to pad the sim_width
        output = da.stack([da.ones_like(lens.profile)] * n_pixels_output) * output_medium
        arr = da.vstack([arr, profile, output])

    return arr


def check_simulations_are_stackable(paths):

    # check if simulations have the same dimensions

    shapes = []
    for p in paths:
        sim = load_full_sim_propagation_v3(p)
        shapes.append(sim.shape)
    
    return np.allclose(shapes, shapes[0])


def load_multi_simulations(paths: list):
    """Load multiple simulations and stack them ontop of each other"""
    data = []
    stackable = check_simulations_are_stackable(paths)
    for p in paths:
        sim = load_full_sim_propagation_v3(p)
        data.append(sim)

    mega = da.hstack(data)

    return mega



def load_full_sim_propagation_v3(path):
    """Dask version"""
    metadata = utils.load_metadata(path)
    n_stages = len(metadata["stages"]) + 1
    sim_paths = [os.path.join(path, str(i), "sim.zarr") for i in range(n_stages)]
    
    import dask.array as da

    data = []
    for sim_path in sim_paths:
        data.append(da.from_zarr(sim_path))
    
    sim = da.vstack(data)

    return sim


def load_full_sim_propagation_v2(path):
    # TODO: move to utils..
    metadata = utils.load_metadata(path)
    n_stages = len(metadata["stages"]) + 1
    sim_paths = [os.path.join(path, str(i), "sim.zarr") for i in range(n_stages)]

    full_sim = None

    for sim_path in sim_paths:
        sim = utils.load_simulation(sim_path)

        if full_sim is None:
            full_sim = sim
        else:
            full_sim = np.vstack([full_sim, sim])

    return full_sim

def plot_sim_propagation_v2(path: Path, axis:int = 1, prop: float = 0.5, log: bool = False, transpose: bool = True) -> tuple:

    full_sim = load_full_sim_propagation_v2(path)
    view = slice_simulation_view(full_sim, axis=axis, prop=prop)
    
    if log:
        view = np.log(view + 1e-12)

    if transpose:
        view = view.T
        figsize = (6, 3)
    else:
        figsize = (3, 6)

    view_fig = plt.figure(figsize=figsize)
    plt.imshow(view, cmap="turbo", aspect="auto")
    plt.colorbar()
    if axis == 1:
        plt.title("Top Down View")
    if axis == 2:
        plt.title("Side On View")

    # TODO: propagation distances as extent

    return view_fig

def save_propagation_gif_full(path: str):

    sim = load_full_sim_propagation_v2(path)

    save_path = os.path.join(path, "propagation.gif")
    imageio.mimsave(save_path, sim, duration=0.2)


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

import dask.array as da

def create_3d_lens(lens: Lens) -> da.Array:
    """Convert the 2D lens height map into 3D profile"""
    # ref: https://stackoverflow.com/questions/59851954/numpy-use-2d-array-as-heightmap-like-index-for-3d-array

    # TODO: figure out why dask is incredibly slow here...

    # scale profile by pixelsize
    lens_profile = lens.profile / lens.pixel_size # profile height in pixels

    l_max = int(np.max(lens_profile))
    arr3d = np.ones(shape=(l_max, lens_profile.shape[0], lens_profile.shape[1]))

    for y in range(lens.profile.shape[0]):
        for x in range(lens.profile.shape[1]):
            height = int(lens_profile[y, x])
            arr3d[height:, y, x] = 0

    # apply aperture
    arr3d[:, lens.aperture == 1] = 0

    return da.from_array(arr3d)

def view_lens3d_in_napari(arr3d: np.ndarray) -> None:
    """View the 3D lens in napari viewer"""
    # create a viewer and add some images
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(arr3d, name="lens", colormap="gray", rendering="iso", scale=[1, 1, 1], depiction="volume")

    napari.run()
