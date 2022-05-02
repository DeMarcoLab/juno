from lib2to3.pgen2.literals import simple_escapes
import os
import uuid

import petname
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from pytest import param
from scipy import fftpack
from tqdm import tqdm

from lens_simulation import utils
from lens_simulation.Lens import Lens, LensType
from lens_simulation.Medium import Medium
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
            lens_type=LensType[self.config["sim_parameters"]["lens_type"]]
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

                    sim_stage.lens.generate_profile(self.parameters.pixel_size, lens_type=self.parameters.lens_type)
                    sim_stage.lens.invert_profile()
                    sim_stage.lens_inverted = True

            if sim_stage.options["use_equivalent_focal_distance"]:
                eq_fd = calculate_equivalent_focal_distance(
                    sim_stage.lens, sim_stage.output
                )
                # print("EQ_FD: ", eq_fd)

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

            # TODO: remove
            self.stage_id = stage_id
            self.progress_bar.set_description(
                f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Propagating Wavefront"
            )
            propagation = self.propagate_wavefront(
                stage, passed_wavefront=passed_wavefront
            )

            if self.options.save:
                self.progress_bar.set_description(
                    f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Saving Simulation"
                )
                # self.save_simulation(sim, stage_id)

            if self.options.save_plot:
                self.progress_bar.set_description(
                    f"Sim: {self.petname} ({str(self.sim_id)[-10:]}) - Plotting Simulation"
                )


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
                pixel_size=self.parameters.pixel_size,
                lens_type=self.parameters.lens_type
            )

        return lens_dict

    def propagate_wavefront(
        self, sim_stage: SimulationStage, passed_wavefront=None,
    ):

        lens = sim_stage.lens
        output_medium = sim_stage.output
        n_slices = sim_stage.n_slices
        start_distance = sim_stage.start_distance
        finish_distance = sim_stage.finish_distance

        DEBUG = self.options.debug

        # padding (width of lens on each side)
        pad_px = lens.profile.shape[-1]
        sim_profile = pad_simulation(lens, pad_px=pad_px)

        # generate frequency array
        freq_arr = generate_sq_freq_arr(
            sim_profile, pixel_size=self.parameters.pixel_size
        )

        # calculate delta and phase profiles
        delta = calculate_delta_profile(
            sim_profile=sim_profile, lens=lens, output_medium=output_medium
        )
        phase = calculate_phase_profile(
            delta=delta, wavelength=self.parameters.sim_wavelength
        )

        A = self.parameters.A if passed_wavefront is None else 1.0
        wavefront = calculate_wavefront(
            phase=phase, passed_wavefront=passed_wavefront, A=A, pad_px=pad_px
        )

        # fourier transform of wavefront
        fft_wavefront = fftpack.fft2(wavefront)

        # pre-allocate view arrays
        sim = np.zeros(shape=(n_slices, *sim_profile.shape), dtype=np.float32)
        top_down_view = np.zeros(
            shape=(n_slices, sim_profile.shape[1]), dtype=np.float32
        )
        side_on_view = np.zeros(
            shape=(n_slices, sim_profile.shape[0]), dtype=np.float32
        )

        if DEBUG:
            print(f"sim_profile.shape={sim_profile.shape}")
            print(f"freq_arr.shape={freq_arr.shape}")
            print(f"delta.shape={delta.shape}")
            print(f"phase.shape={phase.shape}")
            print(f"wavefront.shape={wavefront.shape}")
            print(f"fft_wavefront.shape={fft_wavefront.shape}")
            print(f"top_down_view.shape={top_down_view.shape}")
            print(f"side_on_view.shape={side_on_view.shape}")

            # check the freq arr was created correctly
            assert freq_arr.shape[-1] == wavefront.shape[-1]
            if passed_wavefront is not None:
                assert not np.array_equal(
                    np.unique(wavefront), [0 + 0j]
                )  # non empty sim

        # propagate the wavefront over distance
        distances = np.linspace(start_distance, finish_distance, n_slices)

        prop_progress_bar = tqdm(distances, leave=False)
        for i, distance in enumerate(prop_progress_bar):
            prop_progress_bar.set_description(
                f"Propagating Wavefront at Distance {distance:.4f} / {distances[-1]:.4f}m"
            )

            rounded_output, propagation = propagate_over_distance(
                fft_wavefront, distance, freq_arr, output_medium.wave_number
            )

            if self.options.save:
                # save output
                utils.save_simulation_slice(rounded_output, 
                    fname=os.path.join(self.log_dir, str(self.stage_id), f"{distance*1000:.8f}mm.npy")
                    )
            
            if lens.profile.ndim == 2:
                # calculate views
                centre_px_h = rounded_output.shape[0] // 2
                centre_px_v = rounded_output.shape[1] // 2
                top_down_slice = rounded_output[centre_px_v, :]
                side_on_slice = rounded_output[:, centre_px_h]

                # append views
                top_down_view[i, :] = top_down_slice
                side_on_view[i, :] = side_on_slice
                sim[i, :, :] = rounded_output
            else:
                top_down_view[i, :] = rounded_output


        if self.options.save:
            self.save_simulation(sim, self.stage_id)


        # TODO: separate plotting / save from simulating
        ################## SAVE ##################
        if lens.profile.ndim == 2:
            utils.plot_image(freq_arr, "Frequency Array", 
                    save=True, fname=os.path.join(self.log_dir, str(self.stage_id), "freq.png"))

            utils.plot_image(delta, "Delta Profile", 
                save=True, fname=os.path.join(self.log_dir, str(self.stage_id), "delta.png"))

            utils.plot_image(phase, "Phase Profile", 
                    save=True, fname=os.path.join(self.log_dir, str(self.stage_id), "phase.png"))

        if self.options.save_plot:
            # save top-down
            fig = utils.plot_simulation(
                arr=top_down_view,
                pixel_size_x=self.parameters.pixel_size,
                start_distance=start_distance,
                finish_distance=finish_distance,
            )

            utils.save_figure(
                fig, os.path.join(self.log_dir, str(self.stage_id), "topdown.png")
            )
            plt.close(fig)


            fig = utils.plot_simulation(
                np.log(top_down_view + 10e-12),
                pixel_size_x=self.parameters.pixel_size,
                start_distance=start_distance,
                finish_distance=finish_distance,
            )

            utils.save_figure(
                fig, os.path.join(self.log_dir, str(self.stage_id), "log_topdown.png")
            )
            plt.close(fig)

            fig = utils.plot_simulation(
                arr=side_on_view,
                pixel_size_x=self.parameters.pixel_size,
                start_distance=start_distance,
                finish_distance=finish_distance,
            )
            utils.save_figure(
                fig, os.path.join(self.log_dir, str(self.stage_id), "sideon.png")
            )
            plt.close(fig)

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


def generate_sq_freq_arr(sim_profile: np.ndarray, pixel_size: float) -> np.ndarray:
    """Generate the squared frequency array for the simulation"""
    
    if sim_profile.ndim != 2:
        raise TypeError(f"Only 2D Simulation Profile is supported. Simulation profile of shape {sim_profile.shape} not supported.")

    if sim_profile.shape[0] == 1: # 1D lens
        freq_arr = generate_squared_frequency_array(
            n_pixels=sim_profile.shape[1], pixel_size=pixel_size
        )
    else: # 2D lens
        x = generate_squared_frequency_array(sim_profile.shape[1], pixel_size)
        y = generate_squared_frequency_array(sim_profile.shape[0], pixel_size)
        X, Y = np.meshgrid(x, y)
        freq_arr = X + Y


    return freq_arr.astype(np.float32)


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


def pad_simulation_old(lens: Lens, pad_px: tuple = None) -> np.ndarray:
    """Pad the area around the lens profile to prevent reflection"""

    if lens.profile.ndim != 2:
        raise TypeError(
            f"Pad simulation only supports two-dimensional lens. Lens shape was: {lens.profile.shape}."
        )

    if pad_px is None:
        if lens.profile.ndim == 2:
            pad_px = lens.profile.shape  # TODO: check symmetry

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

    sim_profile = np.pad(
        lens.profile, ((vpad_px, vpad_px), (hpad_px, hpad_px)), mode="constant"
    )

    # TODO: pad symmetrically

    return sim_profile


def pad_simulation(lens: Lens, pad_px: int = None) -> np.ndarray:
    """Pad the area around the lens profile to prevent reflection"""

    # TODO: make this work for assymmetric shape
    if lens.profile.ndim not in (1, 2):
        raise TypeError(
            f"Padding is only supported for 1D and 2D lens. Lens shape was: {lens.profile.shape}."
        )

    if pad_px is None:
        pad_px = lens.profile.shape[-1]

    # two different types of padding?
    sim_profile = np.pad(lens.profile, pad_px, mode="constant")

    if lens.profile.ndim == 1:
        sim_profile = np.expand_dims(
            sim_profile, axis=0
        )  # expand 1D lens to 2D sim shape

    return sim_profile


def calculate_delta_profile(
    sim_profile, lens: Lens, output_medium: Medium
) -> np.ndarray:
    """Calculate the delta profile of the wave"""
    delta = (
        lens.medium.refractive_index - output_medium.refractive_index
    ) * sim_profile

    return delta


def calculate_phase_profile(delta: np.ndarray, wavelength: float) -> np.ndarray:
    """Calculate the phase profile of the wave"""
    phase = 2 * np.pi * delta / wavelength  # % (2 * np.pi)

    return phase


def calculate_wavefront(
    phase: np.ndarray, passed_wavefront: np.ndarray, A: float, pad_px: int = 0
) -> np.ndarray:
    """Calculate the wavefront of light"""

    # only amplifiy the first stage propagation
    if passed_wavefront is not None:
        assert A == 1, "Amplitude should be 1.0. Only amplify the first stage."
        wavefront = A * np.exp(1j * phase) * passed_wavefront
    else:
        wavefront = A * np.exp(1j * phase)

    # padded area should be 0+0j
    if passed_wavefront is not None:
        wavefront[phase == 0] = 0 + 0j

    # zero out padded area (TODO: replace with apeture mask)
    if pad_px:
        wavefront[:, :pad_px] = 0 + 0j
        wavefront[:, -pad_px:] = 0 + 0j

    return wavefront

def propagate_over_distance(
    fft_wavefront, distance, freq_arr, wave_number
) -> np.ndarray:
    prop = np.exp(1j * wave_number * distance) * np.exp(
        (-1j * 2 * np.pi ** 2 * distance * freq_arr) / wave_number
    )
    propagation = fftpack.ifft2(prop * fft_wavefront)

    output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2) ** 2

    rounded_output = np.round(output.astype(np.float32), 10)

    return rounded_output, propagation