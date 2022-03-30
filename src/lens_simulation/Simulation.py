import numpy as np
from scipy import fftpack
import os
from lens_simulation.Lens import Lens, Medium
import uuid
from typing import Optional

import petname

import matplotlib.pyplot as plt
from lens_simulation import Lens, Simulation
from scipy import fftpack
from lens_simulation import utils


class Simulation:
    def __init__(self, config: dict) -> None:

        self.sim_id = uuid.uuid4()
        self.petname = petname.Generate(2)  # TODO: maybe
        self.read_configuration(config=config)
        self.setup_simulation()

    def read_configuration(self, config):

        # TODO: add a check to the config
        self.config = config
        self.run_id = config["run_id"]
        self.parameters = config["parameters"]

    def setup_simulation(self):

        self.log_dir = os.path.join(self.config["log_dir"], str(self.sim_id))
        os.makedirs(self.log_dir, exist_ok=True)

    def run_simulation(self):
        print("-" * 50)
        print(f"Running Simulation {self.petname} ({str(self.sim_id)[-10:]})")
        print(f"Parameters:  {self.parameters}")

        # TODO: actually run the simulation
        # generate_lens_profile
        # generate_medium_mesh
        # generate_frequency_array
        # generate_differential_refractive_profile
        # internal_lens_propagation
        # free_space_propagation

        print("HELLO")


        # lens creation
        amplitude = 10000
        sim_width = 4500e-6
        pixel_size = 1e-6 
        n_slices = 1000

        lens_1 = Lens.Lens(
            diameter=sim_width,
            height=70e-6,
            exponent=0.,
            medium=Lens.Medium(2.348)
        )
        lens_1.generate_profile(pixel_size=pixel_size)

        centre_px = (len(lens_1.profile)-1)//2

        plt.plot(lens_1.profile)
        plt.show()

        # lens 2 creation
        amplitude = 10000
        sim_width = 4500e-6
        pixel_size = 1e-6 

        lens_2 = Lens.Lens(
            diameter=sim_width,
            height=70e-6,
            exponent=2.,
            medium=Lens.Medium(2.348)
        )
        lens_2.generate_profile(pixel_size=pixel_size)

        centre_px = (len(lens_2.profile)-1)//2

        plt.plot(lens_2.profile)
        plt.show()

        # simulation Parameters
        A = 10000
        output_medium_1 = Lens.Medium(2.348)
        output_medium_2 = Lens.Medium(1.5)
        sim_wavelength = 488e-9
        step_size = 0.1e-6



        # Simulation Calculations
        # lens 1 



        def propagate_wavefront(lens, output_medium, n_slices, start_distance, finish_distance):
            print("Lens 1")
            frequency_array_1 = generate_squared_frequency_array(
                n_pixels=len(lens.profile), pixel_size=pixel_size)

            print(len(frequency_array_1))

            delta_1 = (lens.medium.refractive_index-output_medium.refractive_index) * lens.profile
            phase_1 = (2 * np.pi * delta_1 / sim_wavelength) % (2 * np.pi)
            wavefront_1 = A * np.exp(1j * phase_1)
            wavefront_1 = fftpack.fft(wavefront_1)

            start_distance = 0.
            finish_distance = 10.e-3


            sim = np.ones(shape=((n_slices), len(lens.profile)))
            distances_1 = np.linspace(start_distance, finish_distance, n_slices)
            for i, z in enumerate(distances_1):
                prop_1 = np.exp(1j * output_medium.wave_number * z) * np.exp(
                    (-1j * 2 * np.pi ** 2 * z * frequency_array_1) / output_medium.wave_number
                )
                # print("prop shape: ", prop.shape)
                propagation = fftpack.ifft(prop_1 * wavefront_1)

                output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

                sim[i] = np.round(output, 10)
            from lens_simulation import utils
            utils.plot_simulation(sim, sim.shape[1], sim.shape[0], 
                                    pixel_size, start_distance, finish_distance)
            # plt.imshow(sim)
            plt.show()

            return propagation
        
        propagation = propagate_wavefront(lens=lens_1, output_medium=output_medium_1, 
                    n_slices=100, start_distance=0., finish_distance=10.e-3)

        passed_wavefront = propagation

        # lens 2
        print("Lens 2")
        frequency_array_2 = generate_squared_frequency_array(
            n_pixels=len(lens_2.profile), pixel_size=pixel_size)

        print(len(frequency_array_2))

        delta_2 = (lens_2.medium.refractive_index-output_medium_2.refractive_index) * lens_2.profile
        phase_2 = (2 * np.pi * delta_2 / sim_wavelength) % (2 * np.pi)
        wavefront_2 = A * np.exp(1j * phase_2) * passed_wavefront
        fft_wavefront_2 = fftpack.fft(wavefront_2)

        equivalent_focal_distance_2 = calculate_equivalent_focal_distance(lens_2, output_medium_2)
        start_distance_2 = 0 * equivalent_focal_distance_2
        finish_distance_2 = 2 * equivalent_focal_distance_2

        n_slices_2 = 1000

        sim_2 = np.ones(shape=((n_slices_2), len(lens_2.profile)))
        distances_2 = np.linspace(start_distance_2, finish_distance_2, n_slices_2)
        for i, z in enumerate(distances_2):
            prop_2 = np.exp(1j * output_medium_2.wave_number * z) * np.exp(
                (-1j * 2 * np.pi ** 2 * z * frequency_array_2) / output_medium_2.wave_number
            )
            # print("prop shape: ", prop.shape)
            propagation = fftpack.ifft(prop_2 * fft_wavefront_2)

            output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

            sim_2[i] = np.round(output, 10)

        passed_wavefront = propagation

        from lens_simulation import utils
        utils.plot_simulation(sim_2, sim_2.shape[1], sim_2.shape[0], pixel_size, start_distance_2, finish_distance_2)
        # plt.imshow(sim)
        plt.show()

        # output 2
        print("Output 2")
        frequency_array_2 = generate_squared_frequency_array(
            n_pixels=len(lens_2.profile), pixel_size=pixel_size)

        print(len(frequency_array_2))

        delta_2 = (lens_2.medium.refractive_index-output_medium_2.refractive_index) * lens_2.profile
        phase_2 = (2 * np.pi * delta_2 / sim_wavelength) % (2 * np.pi)
        wavefront_2 = A * np.exp(1j * phase_2) * passed_wavefront
        fft_wavefront_2 = fftpack.fft(wavefront_2)

        equivalent_focal_distance_2 = calculate_equivalent_focal_distance(lens_2, output_medium_2)
        start_distance_2 = 0 * equivalent_focal_distance_2
        finish_distance_2 = 2 * equivalent_focal_distance_2

        n_slices_2 = 1000

        sim_2 = np.ones(shape=((n_slices_2), len(lens_2.profile)))
        distances_2 = np.linspace(start_distance_2, finish_distance_2, n_slices_2)
        for i, z in enumerate(distances_2):
            prop_2 = np.exp(1j * output_medium_2.wave_number * z) * np.exp(
                (-1j * 2 * np.pi ** 2 * z * frequency_array_2) / output_medium_2.wave_number
            )
            # print("prop shape: ", prop.shape)
            propagation = fftpack.ifft(prop_2 * fft_wavefront_2)

            output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

            sim_2[i] = np.round(output, 10)

        passed_wavefront = propagation

        utils.plot_simulation(sim_2, sim_2.shape[1], sim_2.shape[0], pixel_size, start_distance_2, finish_distance_2)
        # plt.imshow(sim)
        plt.show()

        # lens 3
        print("Lens 3")
        frequency_array_2 = generate_squared_frequency_array(
            n_pixels=len(lens_2.profile), pixel_size=pixel_size)

        print(len(frequency_array_2))

        delta_2 = (lens_2.medium.refractive_index-output_medium_2.refractive_index) * lens_2.profile
        phase_2 = (2 * np.pi * delta_2 / sim_wavelength) % (2 * np.pi)
        wavefront_2 = A * np.exp(1j * phase_2) * passed_wavefront
        fft_wavefront_2 = fftpack.fft(wavefront_2)

        equivalent_focal_distance_2 = calculate_equivalent_focal_distance(lens_2, output_medium_2)
        start_distance_2 = 0 * equivalent_focal_distance_2
        finish_distance_2 = 2 * equivalent_focal_distance_2

        n_slices_2 = 1000

        sim_2 = np.ones(shape=((n_slices_2), len(lens_2.profile)))
        distances_2 = np.linspace(start_distance_2, finish_distance_2, n_slices_2)
        for i, z in enumerate(distances_2):
            prop_2 = np.exp(1j * output_medium_2.wave_number * z) * np.exp(
                (-1j * 2 * np.pi ** 2 * z * frequency_array_2) / output_medium_2.wave_number
            )
            # print("prop shape: ", prop.shape)
            propagation = fftpack.ifft(prop_2 * fft_wavefront_2)

            output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

            sim_2[i] = np.round(output, 10)

        passed_wavefront = propagation

        utils.plot_simulation(sim_2, sim_2.shape[1], sim_2.shape[0], pixel_size, start_distance_2, finish_distance_2)
        # plt.imshow(sim)
        plt.show()









        print("-" * 50)


def generate_squared_frequency_array(n_pixels: int, pixel_size: float) -> np.ndarray:
    """Generates the squared frequency array used in the fresnel diffraction integral

    Parameters
    ----------
    n_pixels : int
        number of pixels in the lens array
    pixel_size : float
        realspace size of each pixel in the lens array

    Returns
    -------
    np.ndarray
        squared frequency array
    """
    return np.power(fftpack.fftfreq(n_pixels, pixel_size), 2)


def calculate_equivalent_focal_distance(lens: Lens, medium: Medium) -> float:
    """Calculates the focal distance of a lens with any exponent as if it had
    an exponent of 2.0 (assuming plano-concave focusing lens)

    Parameters
    ----------
    lens : Lens
        Lens instance as defined in Lens.py
    medium : Medium
        Medium instance as defined in Lens.py

    Returns
    -------
    float
        returns the calculated focal distance
    """
    equivalent_radius_of_curvature = 0.5 * (
        lens.height + (lens.diameter / 2) ** 2 / lens.height
    )
    equivalent_focal_distance = (
        equivalent_radius_of_curvature * medium.refractive_index
    ) / (lens.medium.refractive_index - medium.refractive_index)

    return equivalent_focal_distance
