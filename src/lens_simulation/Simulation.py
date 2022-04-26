import os
import uuid

import petname
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from scipy import fftpack
from tqdm import tqdm

from lens_simulation import utils
from lens_simulation.Lens import Lens, Medium
from lens_simulation.SimulationUtils import (
    SimulationOptions,
    SimulationParameters,
    SimulationStage,
    SimulationRun,
)

# DONE:
# sweepable parameters
# database management
# visualisation, analytics, comparison

# TODO:
# TODO: initial beam definition (tilt, convergence, divergence)
# TODO: user interface
# TODO: tools (cleaning, sheet measurement, validation, lens creation)
# total internal reflection check (exponential profile)
# TODO: performance (cached results, gpu, parallelism)


#### 2D Simulations ####
# Add support for 2D lens simulations.
# 2D refers to the lens profile, which creates a 3D simulation (volume).
#
# DONE
# convert sim to 2d (1, n)
# update visualisation for 3d volumes (slice through, default to midpoint horizontally)
# convert sim to work for 2d (n, n)
# refactor lens profile to support 2D

# TODO:
# update viz for vertical slicing, and user defined slice plane
# refactor names for 3-d axes
# update and add tests


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

        # options
        self.options = SimulationOptions(
            save=config["options"]["save"],
            save_plot=config["options"]["save_plot"],
            verbose=config["options"]["verbose"],
            debug=config["options"]["debug"],
        )

    def setup_simulation(self):

        self.log_dir = os.path.join(
            self.config["log_dir"], str(self.sim_id)
        )  # TODO: change to petname? careful of collisions
        os.makedirs(self.log_dir, exist_ok=True)

        # common sim parameters
        self.parameters = SimulationParameters(
            A=self.config["sim_parameters"]["A"],
            pixel_size=self.config["sim_parameters"]["pixel_size"],
            sim_width=self.config["sim_parameters"]["sim_width"],
            sim_wavelength=self.config["sim_parameters"]["sim_wavelength"],
        )

        # generate all mediums for simulation
        self.medium_dict = self.generate_mediums()

        # generate all lenses for the simulations
        self.lens_dict = self.generate_lenses()

        # generate all simulation stages
        self.generate_simulation_stages()

        # TODO: figure out how to implement this better
        # self.sim_run = SimulationRun(
        #     id = self.sim_id,
        #     petname = self.petname,
        #     parameters = self.parameters,
        #     config=self.config,
        #     options = self.options,
        #     stages = self.sim_stages
        # )

        # pprint(self.sim_run)

    def generate_simulation_stages(self):

        # validate all lens, mediums exist?
        for stage in self.stages:
            assert (
                stage["output"] in self.medium_dict
            ), f"{stage['output']} has not been defined in the configuration"
            assert (
                stage["lens"] in self.lens_dict
            ), f"{stage['lens']} has not been defined in the configuration"

            assert "n_slices" in stage, f"Stage requires n_slices"
            assert "start_distance" in stage, f"Stage requires start_distance"
            assert "finish_distance" in stage, f"Stage requires finish_distance"

        self.sim_stages = []

        for i, stage in enumerate(self.stages):

            sim_stage = SimulationStage(
                lens=self.lens_dict[stage["lens"]],
                output=self.medium_dict[stage["output"]],
                n_slices=stage["n_slices"],
                start_distance=stage["start_distance"],
                finish_distance=stage["finish_distance"],
                options=stage["options"],
                lens_inverted=False,
            )

            # TODO: determine the best way to do double sided lenses (and define them in the config?)
            # TODO: should we separate double sided lens from inverting?
            if i != 0:

                # NOTE: if the lens and the output have the same medium, the lens is assumed to be 'double-sided'
                # therefore, we invert the lens profile to create an 'air lens' to properly simulate the double sided lens

                if (
                    sim_stage.lens.medium.refractive_index
                    == sim_stage.output.refractive_index
                ):  # TODO: figure out why dataclass comparison isnt working

                    if (
                        sim_stage.lens.medium.refractive_index
                        == self.sim_stages[i - 1].output.refractive_index
                    ):
                        raise ValueError(
                            "Lens and Medium on either side are the same Medium, Lens has no effect."
                        )  # TODO: might be useful for someone...

                    # change to 'air' lens, and invert the profile
                    sim_stage.lens = Lens(
                        diameter=self.parameters.sim_width,
                        height=sim_stage.lens.height,
                        exponent=sim_stage.lens.exponent,
                        medium=self.sim_stages[i - 1].output,
                    )  # replace the lens with lens of previous output medium

                    sim_stage.lens.generate_profile(self.parameters.pixel_size)
                    sim_stage.lens.invert_profile()
                    sim_stage.lens_inverted = True

            if sim_stage.options["use_equivalent_focal_distance"]:
                eq_fd = calculate_equivalent_focal_distance(
                    sim_stage.lens, sim_stage.output
                )

                sim_stage.start_distance = 0.0 * eq_fd
                sim_stage.finish_distance = (
                    sim_stage.options["focal_distance_multiple"] * eq_fd
                )

                # update the metadata if this option is used...
                self.config["stages"][i]["start_distance"] = sim_stage.start_distance
                self.config["stages"][i]["finish_distance"] = sim_stage.finish_distance

            # update config
            self.config["stages"][i]["lens_inverted"] = sim_stage.lens_inverted

            self.sim_stages.append(sim_stage)

    def run_simulation(self):

        # Simulation Calculations
        passed_wavefront = None
        self.progress_bar = tqdm(self.sim_stages, leave=False)
        for stage_id, stage in enumerate(self.progress_bar):

            self.progress_bar.set_description(
                f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Propagating Wavefront"
            )
            sim, propagation = self.propagate_wavefront(
                stage, passed_wavefront=passed_wavefront
            )

            if self.options.save:
                self.progress_bar.set_description(
                    f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Saving Simulation"
                )
                self.save_simulation(sim, stage_id)

            if self.options.save_plot:
                self.progress_bar.set_description(
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
                    self.parameters.pixel_size,
                    stage.start_distance,
                    stage.finish_distance,
                )

                utils.save_figure(
                    fig, os.path.join(self.log_dir, str(stage_id), "img.png")
                )

                plt.close(fig)

            # pass the wavefront to the next stage
            passed_wavefront = propagation

        if self.options.verbose:
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
                diameter=self.parameters.sim_width,
                height=lens["height"],
                exponent=lens["exponent"],
                medium=self.medium_dict[lens["medium"]],
            )

            lens_dict[lens["name"]].generate_profile(
                pixel_size=self.parameters.pixel_size
            )

        return lens_dict

    def propagate_wavefront(
        self, sim_stage: SimulationStage, passed_wavefront=None,
    ):

        # TODO: docstring
        # TODO: input validation

        lens = sim_stage.lens
        output_medium = sim_stage.output
        n_slices = sim_stage.n_slices
        start_distance = sim_stage.start_distance
        finish_distance = sim_stage.finish_distance

        DEBUG = self.options.debug

        # TODO: move this somewhere better
        # create 2D lens profile
        lens.extrude_profile(100e-6)
        # lens.revolve_profile()

        # padding (width of lens on each side)
        if lens.profile.ndim == 1:
            sim_profile = np.pad(lens.profile, len(lens.profile), "constant")
            sim_profile = np.expand_dims(
                sim_profile, axis=0
            )  # TODO: remove this once 2D profiles are implemented...
        else:
            sim_profile = pad_simulation(lens)  # 2D only

        if self.options.verbose:
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
            n_pixels=sim_profile.shape[-1], pixel_size=self.parameters.pixel_size
        )  # TODO: confirm correct for 2D

        delta = (
            lens.medium.refractive_index - output_medium.refractive_index
        ) * sim_profile
        phase = (2 * np.pi * delta / self.parameters.sim_wavelength) % (2 * np.pi)

        # only amplifiy the first stage propagation
        A = self.parameters.A if passed_wavefront is None else 1.0
        if passed_wavefront is not None:
            assert A == 1, "Amplitude is wrong"
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

        # QUERY: ^ dont think this is working properly for 2D

        # fourier transform of wavefront
        fft_wavefront = fftpack.fft2(wavefront)  # TODO: change to np

        sim = np.ones(shape=(n_slices, *sim_profile.shape))

        if DEBUG:
            print(f"{sim_profile.shape=}")
            print(f"{freq_arr.shape=}")
            print(f"{delta.shape=}")
            print(f"{phase.shape=}")
            print(f"{wavefront.shape=}")
            print(f"{fft_wavefront.shape=}")
            print(f"{sim.shape=}")

            # check the freq arr was created correctly
            assert freq_arr.shape[-1] == wavefront.shape[-1]
            assert not np.array_equal(np.unique(wavefront), [0 + 0j])  # non empty sim

        distances = np.linspace(start_distance, finish_distance, n_slices)
        prop_progress_bar = tqdm(distances, leave=False)
        for i, z in enumerate(prop_progress_bar):
            prop_progress_bar.set_description(
                f"Propagating Wavefront at Distance {z:.4f} / {distances[-1]:.4f}m"
            )
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


def pad_simulation(lens: Lens, pad_px: tuple = None) -> np.ndarray:
    """Pad the area around the lens profile to prevent reflection"""

    if lens.profile.ndim != 2:
        raise TypeError(
            f"Pad simulation only supports two-dimensional lens. Lens shape was: {lens.profile.shape}."
        )

    if pad_px is None:
        if lens.profile.ndim == 2:
            pad_px = (0, lens.profile.shape[1])  # TODO: check symmetry

    if not isinstance(pad_px, tuple):
        raise TypeError(
            f"Padding pixels should be given as a tuple in the form (vertical_pad_px, horizontal_pad_px). {type(pad_px)} ({pad_px}) was passed."
        )

    vpad_px = pad_px[0]
    hpad_px = pad_px[1]

    # two different types of padding?
    sim_profile = np.pad(
        lens.profile, ((vpad_px, vpad_px), (hpad_px, hpad_px)), mode="constant"
    )

    return sim_profile
