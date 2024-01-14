from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import fftpack, optimize
from scipy.fftpack import fft2

import juno.utils as j_utils
from juno import utils
from juno.juno_custom.elements.Herschel.structures import *
from juno.Lens import Lens, LensType
from juno.Medium import Medium
from juno.Simulation import generate_sq_freq_arr, propagate_over_distance


def generate_sim_lens(
    settings: HerschelSettings, sim_settings: HerschelSimSettings, pixel_size: float
):
    lens_diameter = 2 * sim_settings.initial_lens_radius

    focal_distance = sim_settings.initial_lens_focal_length
    # if settings.z_input is not None:
    #     focal_distance = settings.z_input

    initial_lens = Lens(
        diameter=lens_diameter,
        height=height_from_focal_distance(
            diameter=lens_diameter,
            lens_medium=sim_settings.n_lens_sim,
            output_medium=settings.n_medium_o,
            focal_distance=focal_distance,
        ),
        exponent=sim_settings.initial_lens_exponent,
        medium=Medium(sim_settings.n_lens_sim),
        lens_type=LensType.Cylindrical,
    )

    initial_lens.generate_profile(pixel_size=pixel_size)
    # initial_lens.profile = initial_lens.profile.max() - initial_lens.profile
    # initial_lens.profile *= settings.scale.value

    if sim_settings.sim_width is not None:
        n_pixels = j_utils._calculate_num_of_pixels(sim_settings.sim_width, pixel_size)
        pad_width = (n_pixels - len(initial_lens.profile[0])) // 2
        initial_lens.profile = np.pad(
            initial_lens.profile,
            ((0, 0), (pad_width, pad_width)),
            mode="constant",
            constant_values=0,
        )
        initial_lens_padding = np.zeros_like(initial_lens.profile[0])
        initial_lens_padding[:pad_width] = 1
        initial_lens_padding[-pad_width:] = 1
    else:
        initial_lens_padding = None

    return initial_lens, initial_lens_padding


def generate_sim_wavefront(
    sim_lens: Lens,
    settings: HerschelSettings,
    sim_lens_padding: np.ndarray = None,
    sim_settings: HerschelSimSettings = None,
):
    profile = sim_lens.profile
    delta = delta_map_from_height(
        profile, sim_settings.n_lens_sim, sim_settings.n_medium_o_sim
    )
    phase = phase_map_from_delta(
        delta, sim_settings.wavelength
    )  # * settings.scale.value)
    wavefront = wavefront_from_phase(phase, amplitude=1)
    if sim_lens_padding is not None:
        wavefront[0][sim_lens_padding == 1] = 0 + 0j
    return wavefront


def generate_lens_first_wavefront(
    lenses: HerschelLensesPadded,
    settings: HerschelSettings,
    sim_settings: HerschelSimSettings,
):
    profile = lenses.first.profile
    delta = delta_map_from_height(profile, sim_settings.n_medium_o_sim, settings.n_lens)
    phase = phase_map_from_delta(delta, sim_settings.wavelength)
    wavefront = wavefront_from_phase(phase, amplitude=1)
    wavefront[lenses.first_padding == 1] = 0 + 0j

    return wavefront


def propagate_zero_lens(
    wavefront: np.ndarray,
    settings: HerschelSettings,
    sim_settings: HerschelSimSettings,
    distances: list,
    pixel_size: float,
    previous_wavefront: np.ndarray = None,
):
    # Calculate media properties
    media_wavelength = (
        sim_settings.wavelength / sim_settings.n_medium_o_sim
    )  # * settings.scale.value
    wavenumber = 2 * np.pi / media_wavelength

    # Calculate the wavefront
    if previous_wavefront is not None:
        wavefront *= previous_wavefront

    # Calculate the fft of the wavefront
    fft_wavefront = fft2(wavefront)
    freq_array = generate_sq_freq_arr(wavefront, pixel_size)  # * settings.scale.value)

    # Convert distances to list
    if isinstance(distances, float) or isinstance(distances, int):
        distances = [distances]

    # Create an array to store the output
    output_array = np.zeros(shape=(len(distances), wavefront.shape[1]))
    for i, distance in enumerate(distances):
        output, propagation = calculate(
            fft_wavefront=fft_wavefront,
            distance=distance,
            wavenumber=wavenumber,
            freq_arr=freq_array,
        )
        output_array[i] = output[0]

    return output, propagation, output_array


def propagate_first_lens(
    wavefront: np.ndarray,
    settings: HerschelSettings,
    sim_settings: HerschelSimSettings,
    distances: list,
    pixel_size: float,
    previous_wavefront: np.ndarray = None,
):
    media_wavelength = (
        sim_settings.wavelength / settings.n_lens
    )  # * settings.scale.value
    wavenumber = 2 * np.pi / media_wavelength

    if previous_wavefront is not None:
        print("Previous wavefront found")
        wavefront *= previous_wavefront

    fft_wavefront = fft2(wavefront)
    freq_array = generate_sq_freq_arr(wavefront, pixel_size)  # * settings.scale.value

    if isinstance(distances, float) or isinstance(distances, int):
        distances = [distances]

    # create an array to store the output
    output_array = np.zeros(shape=(len(distances), wavefront.shape[1]))

    for i, distance in enumerate(distances):
        # print(f"Propagating first lens to {distance}m")
        output, propagation = calculate(
            fft_wavefront=fft_wavefront,
            distance=distance,
            wavenumber=wavenumber,
            freq_arr=freq_array,
        )
        output_array[i] = output[0]

    return output, propagation, output_array


def generate_lens_second_wavefront(
    lenses: HerschelLensesPadded,
    settings: HerschelSettings,
    sim_settings: HerschelSimSettings,
):
    profile = lenses.second.profile
    delta = delta_map_from_height(profile, settings.n_lens, sim_settings.n_medium_i_sim)
    phase = phase_map_from_delta(
        delta, sim_settings.wavelength
    )  # * settings.scale.value)
    wavefront = wavefront_from_phase(phase, amplitude=1)
    wavefront[lenses.second_padding == 1] = 0 + 0j

    return wavefront


def propagate_second_lens(
    wavefront: np.ndarray,
    settings: HerschelSettings,
    sim_settings: HerschelSimSettings,
    distances: list,
    pixel_size: float,
    previous_wavefront: np.ndarray = None,
):
    media_wavelength = (
        sim_settings.wavelength / sim_settings.n_medium_i_sim  # * settings.scale.value
    )
    wavenumber = 2 * np.pi / media_wavelength

    if previous_wavefront is not None:
        wavefront *= previous_wavefront

    fft_wavefront = fft2(wavefront)
    freq_array = generate_sq_freq_arr(wavefront, pixel_size)  # * settings.scale.value

    if isinstance(distances, float) or isinstance(distances, int):
        distances = [distances]

    # create an array to store the output
    output_array = np.zeros(shape=(len(distances), wavefront.shape[1]))

    for i, distance in enumerate(distances):
        # print(f"Propagating second lens to {distance}m")
        output, propagation = calculate(
            fft_wavefront=fft_wavefront,
            distance=distance,
            wavenumber=wavenumber,
            freq_arr=freq_array,
        )

        output_array[i] = output[0]

    return output, propagation, output_array


def height_from_focal_distance(
    diameter: float, lens_medium: float, output_medium: float, focal_distance: float
):
    a = 1
    b = -2 * focal_distance * (lens_medium - output_medium) / output_medium
    c = (diameter / 2) ** 2

    root_1 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    root_2 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # print(f"Root 1: {root_1}")
    # print(f"Root 2: {root_2}")

    if root_1 > 0:
        if root_2 > 0:
            # print("Two positive roots found")
            # print(f"Root 1: {root_1}")
            # print(f"Root 2: {root_2}")
            # print(f"Using min: {min(root_1, root_2)}")
            return min(root_1, root_2)
        else:
            # print("One positive root found")
            # print(f"Root 1: {root_1}")
            # print(f"Using root 1")
            return root_1
    else:
        if root_2 > 0:
            # print("One positive root found")
            # print(f"Root 2: {root_2}")
            # print(f"Using root 2")
            return root_2
        else:
            raise ValueError(
                "Negative value encountered in sqrt.  Can't find a lens height to give this focal distance"
            )

    if b**2 - 4 * a * c < 0:
        raise ValueError(
            "Negative value encountered in sqrt.  Can't find a lens height to give this focal distance"
        )
    else:
        return (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)


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



def calculate(fft_wavefront, distance, freq_arr, wavenumber):
    rounded_output, propagation = propagate_over_distance(
        fft_wavefront, distance, freq_arr, wavenumber
    )
    return rounded_output, propagation