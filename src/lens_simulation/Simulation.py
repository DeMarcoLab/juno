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

        self.sim_id = str(uuid.uuid4())
        self.petname = petname.Generate(2)  # TODO: maybe
        self.read_configuration(config=config)
        self.setup_simulation()

    def read_configuration(self, config):

        # TODO: add a check to the config
        self.config = config
        self.config["sim_id"] = self.sim_id
        self.config["petname"] = self.petname
        self.mediums = config["mediums"]
        self.lenses = config["lenses"]
        self.stages = config["stages"]
        self.options = config["options"]

        # options
        self.verbose = bool(self.config["options"]["verbose"])

    def setup_simulation(self):

        self.log_dir = os.path.join(self.config["log_dir"], str(self.sim_id))
        os.makedirs(self.log_dir, exist_ok=True)

        # common sim parameters
        self.A = self.config["sim_parameters"]["A"]
        self.pixel_size = self.config["sim_parameters"]["pixel_size"]
        self.sim_width = self.config["sim_parameters"]["sim_width"]
        self.sim_wavelength = self.config["sim_parameters"]["sim_wavelength"]

        # generate all mediums for simulation
        self.medium_dict = self.generate_mediums()

        # generate all lenses for the simulations
        self.lens_dict = self.generate_lenses()


    def run_simulation(self):
        print("-" * 50)
        print(f"Running Simulation {self.petname} ({str(self.sim_id)[-10:]})")

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

            # TODO: check if lens and output medium are the same
            # assert block["lens"].medium != block["output"], "Lens and Output cannot have the same Medium."


            sim_stages.append(block)

        print(f"Starting Simulation with {len(sim_stages)} stages.")

        # Simulation Calculations
        passed_wavefront = None
        progress_bar = tqdm(sim_stages)
        for block_id, block in enumerate(progress_bar):
            
            progress_bar.set_description(f"Propagating wavefront...")
            sim, propagation = self.propagate_wavefront(
                lens=block["lens"],
                output_medium=block["output"],
                n_slices=block["n_slices"],
                start_distance=block["start_distance"],
                finish_distance=block["finish_distance"],
                passed_wavefront=passed_wavefront
            )

            if self.options["save"]:
                progress_bar.set_description(f"Saving simulation...")
                self.save_simulation(sim, block_id)

            if self.options["save_plot"]:
                # plot sim result
                fig = utils.plot_simulation(
                    sim,
                    sim.shape[1],
                    sim.shape[0],
                    self.pixel_size,
                    block["start_distance"],
                    block["finish_distance"],
                )

                utils.save_figure(fig, os.path.join(self.log_dir, str(block_id), "img.png"))

                if self.options["plot_sim"]:
                    plt.show()

            passed_wavefront = propagation



        if self.verbose:
            print("-"*20)
            print("---------- Summary ----------")
            
            print("---------- Medium ----------")
            pprint(self.medium_dict)

            print("---------- Lenses ----------")
            pprint(self.lens_dict)

            print("---------- Parameters ----------")
            pprint(self.config["sim_parameters"])

            print("---------- Simulation ----------")
            pprint(sim_stages)

            print("---------- Configuration ----------")
            pprint(self.config)

        utils.save_metadata(self.config, self.log_dir)

        print("-" * 50)

    def save_simulation(self, sim, block_id):
        """Save the simulation data"""
        # TODO: save compressed version?
        save_path = os.path.join(self.log_dir, str(block_id), "sim.npy")
        # save_file = os.path.join(save_path, "sim.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, sim)

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
        
        if self.options["plot_lens"]:
            utils.plot_lenses(lens_dict)

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
        if self.verbose:
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

        return sim, propagation


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
