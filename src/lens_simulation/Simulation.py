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
            diameter=sim_width, height=70e-6, exponent=0.0, medium=Lens.Medium(2.348)
        )
        lens_1.generate_profile(pixel_size=pixel_size)

        centre_px = (len(lens_1.profile) - 1) // 2

        plt.title("Lens 1 Profile")
        plt.plot(lens_1.profile)
        plt.show()

        # lens 2 creation
        amplitude = 10000
        sim_width = 4500e-6
        pixel_size = 1e-6
        self.pixel_size = pixel_size
        lens_2 = Lens.Lens(
            diameter=sim_width, height=70e-6, exponent=2.0, medium=Lens.Medium(2.348)
        )
        lens_2.generate_profile(pixel_size=pixel_size)

        centre_px = (len(lens_2.profile) - 1) // 2

        plt.title("Lens 2 Profile")
        plt.plot(lens_2.profile)
        plt.show()

        # simulation Parameters
        A = 10000
        output_medium_1 = Lens.Medium(2.348)
        output_medium_2 = Lens.Medium(1.5)
        sim_wavelength = 488e-9
        step_size = 0.1e-6

        self.A = A
        self.pixel_size = pixel_size
        self.sim_wavelength = sim_wavelength
        
        
        # Simulation Calculations
        # lens 1
        propagation = self.propagate_wavefront(
            lens=lens_1,
            output_medium=output_medium_1,
            n_slices=100,
            start_distance=0.0,
            finish_distance=10.0e-3,
        )

        passed_wavefront = propagation

        # lens 2
        print("Lens 2")
        equivalent_focal_distance_2 = calculate_equivalent_focal_distance(
            lens_2, output_medium_2
        )
        start_distance = 0 * equivalent_focal_distance_2
        finish_distance = 2 * equivalent_focal_distance_2

        propagation = self.propagate_wavefront(
            lens=lens_2,
            output_medium=output_medium_2,
            n_slices=1000,
            start_distance=start_distance,
            finish_distance=finish_distance,
            passed_wavefront=passed_wavefront,
        )

        passed_wavefront = propagation

        # output 2
        print("Output 2")

        propagation = self.propagate_wavefront(
            lens=lens_2,
            output_medium=output_medium_2,
            n_slices=1000,
            start_distance=start_distance,
            finish_distance=finish_distance,
            passed_wavefront=passed_wavefront,
        )

        passed_wavefront = propagation

        # # lens 3
        print("Lens 3")
        propagation = self.propagate_wavefront(
            lens=lens_2,
            output_medium=output_medium_2,
            n_slices=1000,
            start_distance=start_distance,
            finish_distance=finish_distance,
            passed_wavefront=passed_wavefront,
        )

        passed_wavefront = propagation

        print("-" * 50)

    def propagate_wavefront(
        self,
        lens,
        output_medium,
        n_slices,
        start_distance,
        finish_distance,
        passed_wavefront=None,
    ):

        # TODO: docstring
        # TODO: input validation
        print("-" * 20)
        print(f"Propagating Wavefront with Parameters")
        print(f"Lens: {lens}")
        print(f"Medium: {output_medium}")
        print(
            f"Slices: {n_slices}, Start: {start_distance:.2e}m, Finish: {finish_distance:.2e}m"
        )
        print(f"Passed Wavefront: {passed_wavefront is not None}")
        print(f"-" * 20)

        freq_arr = generate_squared_frequency_array(
            n_pixels=len(lens.profile), pixel_size=self.pixel_size
        )

        delta = (
            lens.medium.refractive_index - output_medium.refractive_index
        ) * lens.profile
        phase = (2 * np.pi * delta / self.sim_wavelength) % (2 * np.pi)
        if passed_wavefront is not None:
            wavefront = self.A * np.exp(1j * phase) * passed_wavefront
        else:
            wavefront = self.A * np.exp(1j * phase)

        fft_wavefront = fftpack.fft(wavefront)

        sim = np.ones(shape=((n_slices), len(lens.profile)))
        distances_2 = np.linspace(start_distance, finish_distance, n_slices)
        for i, z in enumerate(distances_2):
            prop = np.exp(1j * output_medium.wave_number * z) * np.exp(
                (-1j * 2 * np.pi ** 2 * z * freq_arr) / output_medium.wave_number
            )
            propagation = fftpack.ifft(prop * fft_wavefront)

            output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

            sim[i] = np.round(output, 10)

        from lens_simulation import utils

        # plot sim result
        utils.plot_simulation(
            sim,
            sim.shape[1],
            sim.shape[0],
            self.pixel_size,
            start_distance,
            finish_distance,
        )
        # plt.imshow(sim)
        plt.show()

        return propagation


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
