import numpy as np
from scipy import fftpack
import os
from lens_simulation.Lens import Lens, Medium
import uuid
from typing import Optional
from pprint import pprint 

import petname

import matplotlib.pyplot as plt
from lens_simulation import Lens
from scipy import fftpack
from lens_simulation import utils

from tqdm import tqdm


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
        self.sim_parameters = config["sim_parameters"]
        self.parameters = config["parameters"]
        self.mediums = config["mediums"]
        self.lenses = config["lenses"]
        self.stages = config["stages"]

    def setup_simulation(self):

        self.log_dir = os.path.join(self.config["log_dir"], str(self.sim_id))
        os.makedirs(self.log_dir, exist_ok=True)

        # common sim parameters
        self.A = self.sim_parameters["A"]
        self.pixel_size = self.sim_parameters["pixel_size"]
        self.sim_width = self.sim_parameters["sim_width"]
        self.sim_wavelength = self.sim_parameters["sim_wavelength"]

        # generate all mediums for simulation
        self.medium_dict = self.generate_mediums()

        # generate all lenses for the simulations
        self.lens_dict = self.generate_lenses()


    def run_simulation(self):
        print("-" * 50)
        print(f"Running Simulation {self.petname} ({str(self.sim_id)[-10:]})")
        print(f"Parameters:  {self.parameters}")

        # simulation setup
        # (lens_1, output_1) -> (lens_2, output_2) -> (lens_2, output_2)
        # lens -> freespace -> lens -> freespace -> lens -> freespace

        # each sim block needs:
        # lens
        # output_medium
        # n_slices(default 1000)
        # start_distance
        # finish_distance
        
        sim_stages = []

        for i, stage in enumerate(self.stages):
            print(f"Setting up simulation stage {i}")
            block = {
                "lens": self.lens_dict[stage["lens"]],
                "output": self.medium_dict[stage["output"]],
                "n_slices": stage["n_slices"], 
                "start_distance": stage["start_distance"],
                "finish_distance": stage["finish_distance"],
                "options": stage["options"]
            }


            if block["options"]["use_equivalent_focal_distance"]:
                eq_fd = calculate_equivalent_focal_distance(block["lens"], 
                                                            block["output"])
                start_distance = 0.0 * eq_fd
                finish_distance = block["options"]["focal_distance_multiple"] * eq_fd

                block["start_distance"] = start_distance
                block["finish_distance"] = finish_distance

            sim_stages.append(block)

        print(f"Starting Simulation with {len(sim_stages)} stages.")

        # Simulation Calculations
        passed_wavefront = None
        for block in sim_stages:
            print(f"Simulating: {block}")

            propagation = self.propagate_wavefront(
                lens=block["lens"],
                output_medium=block["output"],
                n_slices=block["n_slices"],
                start_distance=block["start_distance"],
                finish_distance=block["finish_distance"],
                passed_wavefront=passed_wavefront
            )

            passed_wavefront = propagation

            if block["options"]["save"]:
                # TODO: save data
                pass

            # TODO: checks
            # check if lens and output medium are the same
            # check if equivalent focal distance calc is set

        
        print("-"*20)
        print("---------- Summary ----------")
        
        print("---------- Medium ----------")
        pprint(self.medium_dict)

        print("---------- Lenses ----------")
        pprint(self.lens_dict)

        print("---------- Parameters ----------")
        pprint(self.sim_parameters)

        print("---------- Simulation ----------")
        pprint(sim_stages)

        print("-" * 50)

    def generate_mediums(self):
        """Generate simulation mediums"""

        medium_dict = {}
        for med in self.mediums:
            
            medium_dict[med["name"]] = Lens.Medium(med["refractive_index"])

        return medium_dict

    def generate_lenses(self):
        
        lens_dict = {}
        for lens in self.lenses:
            
            assert lens["medium"] in self.medium_dict, "Lens Medium not found in simulation mediums"

            lens_dict[lens["name"]] = Lens.Lens(
                diameter=self.sim_width, height=lens["height"],
                exponent=lens["exponent"], medium=self.medium_dict[lens["medium"]]
            ) 

            lens_dict[lens["name"]].generate_profile(pixel_size=self.pixel_size)


        # plot lens profiles
        for name, lens in lens_dict.items():
            # fig, ax = plt.Figure()
            plt.title("Lens Profiles")
            plt.plot(lens.profile, label=name)
            plt.legend(loc="best")
            plt.plot()
                
        return lens_dict

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
