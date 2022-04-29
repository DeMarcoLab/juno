from dis import dis
import numpy as np
from scipy import fftpack
import os
from lens_simulation.Lens import Lens, Medium
import uuid
from typing import Optional

import petname


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

        # Simulation Calculations
        passed_wavefront = None
        progress_bar = tqdm(self.sim_stages, leave=False)
        for block_id, block in enumerate(progress_bar):

            progress_bar.set_description(
                f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Propagating Wavefront"
            )
            sim, propagation = self.propagate_wavefront(
                lens=block["lens"],
                output_medium=block["output"],
                n_slices=block["n_slices"],
                start_distance=block["start_distance"],
                finish_distance=block["finish_distance"],
                passed_wavefront=passed_wavefront,
            )

            if self.options["save"]:
                progress_bar.set_description(
                    f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Saving Simulation"
                )
                self.save_simulation(sim, block_id)

            if self.options["save_plot"]:
                progress_bar.set_description(
                    f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Plotting Simulation"
                )

                # plot sim result
                if sim.ndim == 3:
                    width = sim.shape[2]
                    height = sim.shape[0]
                elif sim.ndim == 2:
                    width = sim.shape[1]
                    height = sim.shape[0]
                else:
                    raise ValueError(f"Simulation of {sim.ndim} is not supported")

                fig = utils.plot_simulation(
                    sim,
                    width,
                    height,
                    self.pixel_size,
                    block["start_distance"],
                    block["finish_distance"],
                )

                utils.save_figure(
                    fig, os.path.join(self.log_dir, str(block_id), "img.png")
                )

                plt.close(fig)

            # pass the wavefront to the next stage
            passed_wavefront = propagation

        if self.verbose:
            print("-" * 20)
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

            assert (
                lens["medium"] in self.medium_dict
            ), "Lens Medium not found in simulation mediums"

            lens_dict[lens["name"]] = Lens(
                diameter=self.sim_width,
                height=lens["height"],
                exponent=lens["exponent"],
                medium=self.medium_dict[lens["medium"]],
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

        DEBUG = self.debug

        # padding (width of lens on each side)
        sim_profile = np.pad(lens.profile, len(lens.profile), "constant")

        # 2d # TODO: move this to profile creation?
        sim_profile = np.expand_dims(sim_profile, axis=0)

        # only amplifiy the first stage propagation
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
            n_pixels=sim_profile.shape[-1], pixel_size=self.pixel_size
        )  # TODO: confirm correct for 2D

        delta = (
            lens.medium.refractive_index - output_medium.refractive_index
        ) * sim_profile
        phase = (2 * np.pi * delta / self.sim_wavelength) % (2 * np.pi)

        if passed_wavefront is not None:
            wavefront = A * np.exp(1j * phase) * passed_wavefront
        else:
            wavefront = A * np.exp(1j * phase)

        # padded area should be 0+0j
        if passed_wavefront is not None:  # TODO: convert to apeture mask
            wavefront[phase == 0] = 0 + 0j
        else:
            wavefront[:, : lens.profile.shape[-1]] = 0 + 0j
            wavefront[:, -lens.profile.shape[-1] :] = (
                0 + 0j
            )  # TODO: confirm this is correct for 2d?

        # fourier transform of wavefront
        fft_wavefront = fftpack.fft2(wavefront)  # TODO: change to np

        sim = np.ones(shape=(n_slices, *sim_profile.shape))

        # if DEBUG:
        #     print(f"{sim_profile.shape=}")
        #     print(f"{freq_arr.shape=}")
        #     print(f"{delta.shape=}")
        #     print(f"{phase.shape=}")
        #     print(f"{wavefront.shape=}")
        #     print(f"{fft_wavefront.shape=}")
        #     print(f"{sim.shape=}")

        #     # check the freq arr was created correctly
        #     assert freq_arr.shape[-1] == wavefront.shape[-1]
        #     assert not np.array_equal(np.unique(wavefront), [0 + 0j])  # non empty sim

        distances = np.linspace(start_distance, finish_distance, n_slices)
        for i, z in enumerate(distances):
            prop = np.exp(1j * output_medium.wave_number * z) * np.exp(
                (-1j * 2 * np.pi ** 2 * z * freq_arr) / output_medium.wave_number
            )
            propagation = fftpack.ifft2(prop * fft_wavefront)

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
