import numpy as np
from scipy import fftpack
import os
from lens_simulation.Lens import Lens, Medium
import uuid
from pprint import pprint 

import petname

import matplotlib.pyplot as plt
from scipy import fftpack
from lens_simulation import utils

from tqdm import tqdm

# DONE:
# sweepable parameters
# database management
# visualisation, analytics, comparison

# TODO:
# TODO: initial beam
# TODO: tools:
    # - measuring sheet parameters (full width, half maximum)
    # - cleaning
    # - lens creation (load profile)
    # - total internal reflection check (exponential profile)
# TODO: performance (cached results, gpu)

class Simulation:
    def __init__(self, config: dict) -> None:

        self.sim_id = str(uuid.uuid4())
        self.petname = petname.Generate(2)  
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

        self.log_dir = os.path.join(self.config["log_dir"], str(self.sim_id)) # TODO: change to petname? careful of collisions
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

        # generate all simulation stages
        self.generate_simulation_stages()


    def generate_simulation_stages(self):

        # validate all lens, mediums exist?
        for stage in self.stages:
            assert stage["output"] in self.medium_dict, f"{stage['output']} has not been defined in the configuration"
            assert stage["lens"] in self.lens_dict, f"{stage['lens']} has not been defined in the configuration"
            
            assert "n_slices" in stage, f"Stage requires n_slices"
            assert "start_distance" in stage, f"Stage requires start_distance"
            assert "finish_distance" in stage, f"Stage requires finish_distance"
            
        self.sim_stages = []

        for i, stage in enumerate(self.stages):
           
            block = {
                "lens": self.lens_dict[stage["lens"]],
                "output": self.medium_dict[stage["output"]],
                "n_slices": stage["n_slices"], 
                "start_distance": stage["start_distance"],
                "finish_distance": stage["finish_distance"],
                "options": stage["options"],
                "lens_inverted": False
            }

            # TODO: determine the best way to do double sided lenses (and define them in the config?)
            # TODO: should we separate double sided lens from inverting?
            if i!=0:

                # NOTE: if the lens and the output have the same medium, the lens is assumed to be 'double-sided'
                # therefore, we invert the lens profile to create an 'air lens' to properly simulate the double sided lens

                if block["lens"].medium.refractive_index == block["output"].refractive_index: # TODO: figure out why dataclass comparison isnt working

                    if block["lens"].medium.refractive_index == self.sim_stages[i-1]["output"].refractive_index:
                        raise ValueError("Lens and Medium on either side are the same Medium, Lens has no effect.") # TODO: might be useful for someone...

                    # change to 'air' lens, and invert the profile
                    block["lens"] = Lens(
                        diameter=self.sim_width, 
                        height=block["lens"].height,
                        exponent=block["lens"].exponent, 
                        medium= self.sim_stages[i-1]["output"]
                    ) # replace the lens with lens of previous output medium
                    block["lens"].generate_profile(self.pixel_size)
                    block["lens"].invert_profile()
                    block["lens_inverted"] = True
                                        
                    # TODO: need to update lens config?

                    # assert block["lens"].medium != block["output"], "Lens and Output cannot have the same Medium."

            if block["options"]["use_equivalent_focal_distance"]:
                eq_fd = calculate_equivalent_focal_distance(block["lens"], 
                                                            block["output"])
                start_distance = 0.0 * eq_fd
                finish_distance = block["options"]["focal_distance_multiple"] * eq_fd

                block["start_distance"] = start_distance
                block["finish_distance"] = finish_distance

                # update the metadata if this option is used...
                self.config["stages"][i]["start_distance"] = start_distance
                self.config["stages"][i]["finish_distance"] = finish_distance

            # update config
            self.config["stages"][i]["lens_inverted"] = block["lens_inverted"]

            self.sim_stages.append(block)

    def run_simulation(self):

        # Simulation Calculations
        passed_wavefront = None
        progress_bar = tqdm(self.sim_stages, leave=False)
        for block_id, block in enumerate(progress_bar):
            
            progress_bar.set_description(f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Propagating Wavefront")
            sim, propagation = self.propagate_wavefront(
                lens=block["lens"],
                output_medium=block["output"],
                n_slices=block["n_slices"],
                start_distance=block["start_distance"],
                finish_distance=block["finish_distance"],
                passed_wavefront=passed_wavefront
            )

            if self.options["save"]:
                progress_bar.set_description(f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Saving Simulation")
                self.save_simulation(sim, block_id)

            if self.options["save_plot"]:
                progress_bar.set_description(f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Plotting Simulation")

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
                                
                plt.close(fig)
            
            # pass the wavefront to the next stage
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
            pprint(self.sim_stages)

            print("---------- Configuration ----------")
            pprint(self.config)

        utils.save_metadata(self.config, self.log_dir)

    def save_simulation(self, sim, block_id):
        """Save the simulation data"""
        # TODO: save compressed version?
        save_path = os.path.join(self.log_dir, str(block_id), "sim.npy")
        # save_file = os.path.join(save_path, "sim.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, sim)

    def generate_mediums(self):
        """Generate all the mediums for the simulation"""

        medium_dict = {}
        for med in self.mediums:
            
            medium_dict[med["name"]] = Medium(med["refractive_index"])

        return medium_dict

    def generate_lenses(self):
        """Generate all the lenses for the simulation"""
        lens_dict = {}
        for lens in self.lenses:
            
            assert lens["medium"] in self.medium_dict, "Lens Medium not found in simulation mediums"

            lens_dict[lens["name"]] = Lens(
                diameter=self.sim_width, height=lens["height"],
                exponent=lens["exponent"], medium=self.medium_dict[lens["medium"]]
            ) 

            lens_dict[lens["name"]].generate_profile(pixel_size=self.pixel_size)

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
        
        # padding (width of lens on each side)
        sim_profile = np.pad(lens.profile, len(lens.profile), "constant")

        if passed_wavefront is not None:
            A = 1.0
        else:
            A = self.A

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
            n_pixels=len(sim_profile), pixel_size=self.pixel_size
        )

        delta = (
            lens.medium.refractive_index - output_medium.refractive_index
        ) * sim_profile
        phase = (2 * np.pi * delta / self.sim_wavelength) % (2 * np.pi)

        if passed_wavefront is not None:
            wavefront = A * np.exp(1j * phase) * passed_wavefront
        else:
            wavefront = A * np.exp(1j * phase)

        # padded area should be 0+0j
        if passed_wavefront is not None:
            wavefront[phase == 0] = 0+0j
        else:
            wavefront[:len(lens.profile)] = 0+0j
            wavefront[-len(lens.profile):] = 0+0j

        fft_wavefront = fftpack.fft(wavefront)

        sim = np.ones(shape=((n_slices), len(sim_profile)))
        distances_2 = np.linspace(start_distance, finish_distance, n_slices)
        for i, z in enumerate(distances_2):
            prop = np.exp(1j * output_medium.wave_number * z) * np.exp(
                (-1j * 2 * np.pi ** 2 * z * freq_arr) / output_medium.wave_number
            )
            propagation = fftpack.ifft(prop * fft_wavefront)

            output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

            sim[i] = np.round(output, 10)

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
