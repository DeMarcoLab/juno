import numpy as np
from LightPipes import *
from matplotlib import pyplot as plt
from juno.Simulation import generate_sq_freq_arr, propagate_over_distance
from scipy import fftpack
import yaml

def load_from_yaml(yaml_file):
    with open(yaml_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def get_annular_mask(array, pixel_size, NA_outer, NA_inner, index):
    half_shape = array.shape[0]//2
    x = np.arange(-1, 1+1/half_shape, 1/half_shape)/(2*pixel_size)
    y = x
    [X, Y] = np.meshgrid(x, y)
    R = np.sqrt(X * X + Y * Y)
    MaxRad = NA_outer / index  # maximum annulus diameter
    MinRad = NA_inner / index  # minimum annulus diameter
    AnnularFilter = (R <= MaxRad) & (R >= MinRad)
    return AnnularFilter


### Getting wavefront ###
def wavefront_from_phase(phase: np.ndarray, amplitude: float = 1):
    return amplitude * np.exp(1j * phase)

def wavefront_from_pupil(pupil: np.ndarray):
    return fftpack.fftshift(fftpack.ifft2(pupil))

### Getting delta ###
def delta_map_from_phase(phase_map, wavelength):
    delta_map = phase_map / (2 * np.pi / wavelength)
    return delta_map

def delta_map_from_height(height_map, n_o, n_l):
    delta_map = height_map * (n_o - n_l)
    return delta_map

### Getting phase ###
def phase_map_from_height(height_map, wavelength, n_o, n_l):
    delta_map = (n_o - n_l) * height_map
    phase_map = delta_map * (2 * np.pi / wavelength)
    return phase_map

def phase_map_from_delta(delta_map, wavelength):
    phase_map = delta_map * (2 * np.pi / wavelength)
    return phase_map

### Getting height ###
def height_map_from_delta(delta_map, n_o, n_l):
    height_map = delta_map / (n_o - n_l)
    return height_map

def height_map_from_phase(phase_map, wavelength, n_o, n_l):
    delta_map = delta_map_from_phase(phase_map, wavelength)
    height_map = height_map_from_delta(delta_map, n_o, n_l)
    return height_map

# define a function that takes in any numbers of plots and plots them in a grid
def plot_grid(*args, **kwargs):
    n = len(args)
    # get kwarg title_1
    titles = kwargs.get("titles", None)
    # remove titles from kwargs
    if titles:
        kwargs.pop("titles")
    dpi = kwargs.get("dpi")
    # remove dpi from kwargs
    if dpi:
        kwargs.pop("dpi")
    else:
        dpi = 100
    fig, axes = plt.subplots(1, n, figsize=(n*5, 5))
    for i, (ax, arg) in enumerate(zip(axes, args)):
        plot = ax.imshow(arg, **kwargs)
        if titles:
            # set title
            ax.set_title(titles[i])
        # plt.title(titles[i])
        plt.colorbar(plot, ax=ax)
    plt.tight_layout()
    plt.show()

def calculate_and_plot(fft_wavefront, distance, freq_arr, wavenumber, wavelength, n, cmap, realspace_x, realspace_y, dpi=100):
    rounded_output, propagation = propagate_over_distance(
        fft_wavefront, distance, freq_arr, wavenumber
    )
    plt.figure(figsize=(10, 10), dpi=dpi)
    plt.imshow(rounded_output, cmap=cmap, aspect="auto", extent=[-realspace_x/2, realspace_x/2, -realspace_y/2, realspace_y/2])
    plt.title("Stationary LLS xz PSF {}lambda".format(distance/wavelength))
    plt.xlabel("x/λ")
    plt.ylabel("z/λ")
    plt.colorbar()
    plt.show()
    return rounded_output

def calculate(fft_wavefront, distance, freq_arr, wavenumber):
    rounded_output, propagation = propagate_over_distance(
        fft_wavefront, distance, freq_arr, wavenumber
    )
    return rounded_output, propagation