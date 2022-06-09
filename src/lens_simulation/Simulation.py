import os
from pathlib import Path
import uuid

import petname
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from scipy import fftpack
from tqdm import tqdm

from lens_simulation import utils
from lens_simulation.Lens import Lens, LensType, GratingSettings
from lens_simulation.Medium import Medium
from lens_simulation.structures import (
    SimulationOptions,
    SimulationParameters,
    SimulationStage,
    SimulationResult,
)

from lens_simulation import validation
from lens_simulation.beam import generate_beam

# DONE:
# sweepable parameters
# database management
# visualisation, analytics, comparison
# initial beam definition (tilt, convergence, divergence)

# TODO: tools (cleaning, sheet measurement, validation, lens creation)
# TODO: total internal reflection check (exponential profile)
# TODO: performance (cached results, gpu, parallelism)


class Simulation:
    def __init__(self, config: dict) -> None:

        self.sim_id = str(uuid.uuid4())
        self.petname = petname.Generate(2)
        self.config = self.read_configuration(config=config)
        self.setup_simulation(self.config)

    def read_configuration(self, config):

        config["sim_id"] = self.sim_id
        config["petname"] = self.petname
        config["started"] = utils.current_timestamp()


        # create logging directory
        log_dir = os.path.join(config["log_dir"], str(self.petname))
        os.makedirs(log_dir, exist_ok=True)

        # options
        self.options = SimulationOptions(
            log_dir=log_dir,
            save=config["options"]["save"],
            save_plot=config["options"]["save_plot"],
            verbose=config["options"]["verbose"],
            debug=config["options"]["debug"],
        )
        
        # common sim parameters
        self.parameters = SimulationParameters(
            A=config["sim_parameters"]["A"],
            pixel_size=config["sim_parameters"]["pixel_size"],
            sim_width=config["sim_parameters"]["sim_width"],
            sim_height=config["sim_parameters"]["sim_height"], 
            sim_wavelength=config["sim_parameters"]["sim_wavelength"],
            lens_type=LensType[config["sim_parameters"]["lens_type"]],
        )

        return config

    def setup_simulation(self, config: dict):

        # generate all mediums for simulation
        simulation_mediums = generate_mediums(config["mediums"])

        # generate all lenses for the simulations
        simulation_lenses = generate_lenses(config["lenses"], simulation_mediums, self.parameters)

        # validate sim, lens and medium setup 
        stages_config = validation._validate_simulation_stage_list(config["stages"], simulation_mediums, simulation_lenses)

        # generate all simulation stages
        self.sim_stages = generate_simulation_stages(stages_config, simulation_mediums, simulation_lenses, config, self.parameters)

    def run_simulation(self):
        """Run the simulation propagation over all simulation stages."""

        # NOTE: make functional (need to update SimulationRunner too...?)
        # sim_stages: list, parameters: SimulationParameters, options: SimulationOptions, config: dict
        sim_stages = self.sim_stages
        parameters = self.parameters
        options = self.options
        config = self.config

        petname = config["petname"]
        sim_id = config["sim_id"]

        passed_wavefront = None
        progress_bar = tqdm(sim_stages, leave=False)
        for stage in progress_bar:

            progress_bar.set_description(
                f"Sim: {petname} ({str(sim_id)[-10:]}) - Propagating Wavefront"
            )
            result = propagate_wavefront(
                stage=stage,
                parameters=parameters,
                options=options,
                passed_wavefront=passed_wavefront,
            )

            # save path
            save_path = os.path.join(options.log_dir, str(stage._id))

            if options.save:
                progress_bar.set_description(
                    f"Sim: {petname} ({str(sim_id)[-10:]}) - Saving Simulation"
                )
                utils.save_simulation(result.sim, os.path.join(save_path, "sim.npy"))

            if options.save_plot:
                progress_bar.set_description(
                    f"Sim: {petname} ({str(sim_id)[-10:]}) - Plotting Simulation"
                )

                save_result_plots(result, stage, parameters, save_path)

            # pass the wavefront to the next stage
            passed_wavefront = result.propagation

        # save final sim configruation
        config["finished"] = utils.current_timestamp()
        utils.save_metadata(config, options.log_dir)

def generate_simulation_stages(stages: list, simulation_mediums: dict, simulation_lenses: dict, config: dict, parameters: SimulationParameters) -> list:
    """Generate the list of simulation stages

    Args:
        stages (list): config list containing simulation stages
        simulation_mediums (dict): config dict containing simulation mediums
        simulation_lenses (dict): config dict containing simulation lenses
        config (dict): simulation config 
        parameters (SimulationParameters): simulation parameters

    Returns:
        list[SimulationStage]: list of SimulationStages ready for simulation
    """


    # stages to be simulated
    sim_stages = []

    ################################ BEAM STAGE ################################

    # first stage is a beam
    beam = generate_beam(config["beam"], parameters)
    beam_stage = SimulationStage(
        lens=beam.lens,
        output=Medium(1.33),
        n_slices=beam.settings.n_slices,
        start_distance=beam.start_distance,
        finish_distance=beam.finish_distance,
        tilt={"x": beam.tilt[0], "y": beam.tilt[1]},
    )

    sim_stages.append(beam_stage) 

    ################################ LENS STAGES ################################

    for i, stage in enumerate(stages):

        sim_stage_no = len(sim_stages)
        
        sim_stage = SimulationStage(
            lens=simulation_lenses[stage["lens"]],
            output=simulation_mediums[stage["output"]],
            n_slices=stage["n_slices"],
            step_size=stage["step_size"],
            start_distance=stage["start_distance"],
            finish_distance=stage["finish_distance"],
            options=stage["options"],
            lens_inverted=False,
            _id=sim_stage_no,
        )

        # NOTE: if the lens and the output have the same medium, the lens is assumed to be 'double-sided'
        # therefore, we invert the lens profile to create an 'air lens' to properly simulate the double sided lens
        if (sim_stage.lens.medium.refractive_index == sim_stage.output.refractive_index):  # TODO: figure out why dataclass comparison isnt working

            sim_stage = invert_lens_and_output_medium(sim_stage, sim_stages[sim_stage_no - 1], parameters)

        if sim_stage.options is not None:
            sim_stage = calculate_start_and_finish_distance(sim_stage)
    
        # update config
        config["stages"][i]["start_distance"] = sim_stage.start_distance
        config["stages"][i]["finish_distance"] = sim_stage.finish_distance
        config["stages"][i]["lens_inverted"] = sim_stage.lens_inverted

        # add to simulation
        sim_stages.append(sim_stage)
    
    return sim_stages


def calculate_start_and_finish_distance(stage: SimulationStage):
    if stage.options["use_equivalent_focal_distance"]:
        eq_fd = calculate_equivalent_focal_distance(stage.lens, stage.output)

        stage.start_distance = (stage.options["focal_distance_start_multiple"] * eq_fd)
        stage.finish_distance = (stage.options["focal_distance_multiple"] * eq_fd)
        
    return stage 

def generate_mediums(mediums: list):
    """Generate all the mediums for the simulation"""

    simulation_mediums = {}
    for med in mediums:

        simulation_mediums[med["name"]] = Medium(med["refractive_index"])

    return simulation_mediums

def generate_lenses(lenses: list, simulation_mediums: dict, parameters: SimulationParameters):
    """Generate all the lenses for the simulation"""

    from lens_simulation.Lens import apply_modifications, generate_lens  
    
    simulation_lenses = {}
    for lens_config in lenses:

        if lens_config["medium"] not in simulation_mediums:
            raise ValueError("Lens Medium not found in simulation mediums")

        # generate lens from config
        lens = generate_lens(lens_config=lens_config, medium=simulation_mediums[lens_config["medium"]])

        # check lens fits in the simulation
        if lens.diameter > parameters.sim_width or lens.diameter > parameters.sim_height:
            raise ValueError(
                f"Lens diameter must be smaller than the simulation size: lens: {lens.diameter:.2e}m, sim: {parameters.sim_width:.2e}mx{parameters.sim_height:.2e}m"
            )

        # load a custom lens profile
        if lens_config["custom"]:
            lens.load_profile(fname=lens_config["custom"])

        # generate the profile from the configuration
        else:
            lens.generate_profile(
                pixel_size=parameters.pixel_size,
                lens_type=parameters.lens_type,
                length=lens_config["length"]
            )

        lens = apply_modifications(lens, lens_config, parameters)

        simulation_lenses[lens_config["name"]] = lens

    return simulation_lenses





def propagate_wavefront(
    stage: SimulationStage,
    parameters: SimulationParameters,
    options: SimulationOptions,
    passed_wavefront: np.ndarray = None,
) -> SimulationResult:
    """Propagate the light wavefront using the supplied settings and parameters.

    Args:
        sim_stage (SimulationStage): the setup of the simulation stage, lens -> output
        parameters (SimulationParameters): the global simulation parameters (shared for all stages)
        options (SimulationOptions): global simulation options
        passed_wavefront (np.ndarray, optional): the previous wavefront to propagate from. Defaults to None.

    Raises:
        ValueError: lens is larger than the simulation width

    Returns:
        SimulationResult: results of the wave propagation (including intermediates if debugging)
    """

    lens: Lens = stage.lens
    output_medium: Medium = stage.output
    n_slices: int = stage.n_slices
    step_size: float = stage.step_size    
    start_distance: float = stage.start_distance
    finish_distance: float = stage.finish_distance
    amplitude: float = parameters.A if passed_wavefront is None else 1.0

    DEBUG = options.debug
    save_path = os.path.join(options.log_dir, str(stage._id))

    # pad the lens profile to be the same size as the simulation
    lens = pad_simulation(lens, parameters=parameters)
    
    # apply all aperture masks, TODO: mvoe this to a better place
    lens.apply_aperture_masks()
    sim_profile = lens.profile

    # generate frequency array
    freq_arr = generate_sq_freq_arr(sim_profile, pixel_size=parameters.pixel_size)

    delta = calculate_tilted_delta_profile(sim_profile, lens, output_medium, stage.tilt)

    phase = calculate_phase_profile(delta=delta, wavelength=parameters.sim_wavelength)

    wavefront = calculate_wavefront(
        phase=phase,
        passed_wavefront=passed_wavefront,
        A=amplitude,
        aperture=lens.aperture,
    )

    # fourier transform of wavefront
    fft_wavefront = fftpack.fft2(wavefront)

    # calculate propagation distances
    if n_slices is None:
        n_slices = int((finish_distance - start_distance) / step_size)
    distances = np.linspace(start_distance, finish_distance, n_slices)

    # pre-allocate view arrays
    sim = np.zeros(shape=(n_slices, *sim_profile.shape), dtype=np.float32)
    top_down_view = np.zeros(shape=(n_slices, sim_profile.shape[1]), dtype=np.float32)
    side_on_view = np.zeros(shape=(n_slices, sim_profile.shape[0]), dtype=np.float32)

    # propagate the wavefront over distance
    prop_progress_bar = tqdm(distances, leave=False)
    for i, distance in enumerate(prop_progress_bar):
        prop_progress_bar.set_description(
            f"Propagating Wavefront at Distance {distance:.4f} / {distances[-1]:.4f}m"
        )

        rounded_output, propagation = propagate_over_distance(
            fft_wavefront, distance, freq_arr, output_medium.wave_number
        )

        if options.save:
            # save output
            utils.save_simulation(
                rounded_output,
                fname=os.path.join(save_path, f"{distance*1000:.8f}mm.npy"),
            )

        # calculate views
        centre_px_h = rounded_output.shape[0] // 2
        centre_px_v = rounded_output.shape[1] // 2
        top_down_slice = rounded_output[centre_px_h, :]
        side_on_slice = rounded_output[:, centre_px_v]

        # append views
        top_down_view[i, :] = top_down_slice
        side_on_view[i, :] = side_on_slice
        sim[i, :, :] = rounded_output


    # return results (TODO: reduce in non-debug mode)
    result = SimulationResult(
        propagation=propagation,
        top_down=top_down_view,
        side_on=side_on_view,
        sim=sim,
        sim_profile=sim_profile,
        lens=lens,
        freq_arr=freq_arr,
        delta=delta,
        phase=phase,
    )


    return result


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
        raise TypeError(
            f"Only 2D Simulation Profile is supported. Simulation profile of shape {sim_profile.shape} not supported."
        )

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


def pad_simulation(lens: Lens, parameters: SimulationParameters) -> np.ndarray:
    """Pad the lens profile to match the simulation dimensions. Padding is used to 
        prevent reflection in the simulation

    Args:
        lens (Lens): simulation lens
        parameters (SimulationParameters): simulation parameters

    Raises:
        TypeError: Lens profile is the wrong shape. Only 2D lens are supported.

    Returns:
        np.ndarray: padded lens profile
    """
    if lens.profile.ndim != 2:
        raise TypeError(
            f"Padding is only supported for 2D lens. Lens shape was: {lens.profile.shape}."
        )

    # calculate the number of pixels in the simulation
    sim_n_pixels_height = utils._calculate_num_of_pixels(parameters.sim_height, parameters.pixel_size)
    sim_n_pixels_width = utils._calculate_num_of_pixels(parameters.sim_width, parameters.pixel_size)

    # calculate aperture mask
    lens.sim_aperture_mask  = calculate_sim_aperture(lens, sim_n_pixels_height, sim_n_pixels_width)

    # calculate different in size
    diff_h = (sim_n_pixels_height - lens.profile.shape[0]) // 2
    diff_w = (sim_n_pixels_width - lens.profile.shape[1]) // 2

    # pad the lens profile
    lens.profile = np.pad(lens.profile, pad_width=((diff_h, diff_h), (diff_w, diff_w)), mode="constant")
    
    # apply aperture mask to sim padded area
    # TODO: could change this to calc the mask with == 0?
    lens.profile[lens.sim_aperture_mask] = 0 

    return lens

def calculate_sim_aperture(lens, sim_h, sim_w) -> np.ndarray:
    lens_h, lens_w = lens.profile.shape
    aperture_mask = np.ones(shape=(sim_h, sim_w))
    
    y0 = sim_h//2 - lens_h//2
    y1 = sim_h//2 + lens_h//2
    x0 = sim_w//2 - lens_w//2
    x1 = sim_w//2 + lens_w//2

    aperture_mask[y0:y1, x0:x1] = 0

    return aperture_mask.astype(bool)

def calculate_delta_profile(
    sim_profile: np.ndarray, lens: Lens, output_medium: Medium
) -> np.ndarray:
    """Calculate the delta profile of the wave"""
    delta = (
        lens.medium.refractive_index - output_medium.refractive_index
    ) * sim_profile

    return delta


def calculate_tilted_delta_profile(
    sim_profile: np.ndarray, lens: Lens, output_medium: Medium, tilt: dict = None
) -> np.ndarray:
    """Calculate the delta profile of the wave, and tilt if required.

    Args:
        lens (Lens): lens
        output_medium (Medium): output medium
        tilt: (dict): dictionary containing the titlt values

    Returns:
        np.ndarray: delta profile
    """

    # regular delta calculation
    delta = calculate_delta_profile(sim_profile, lens, output_medium)

    # tilt the beam
    if tilt is not None:
        x = np.arange(len(sim_profile)) * lens.pixel_size
        y = np.arange(len(sim_profile)) * lens.pixel_size

        y_tilt_rad = np.deg2rad(tilt["y"])
        x_tilt_rad = np.deg2rad(tilt["x"])

        # modify the optical path of the light based on tilt
        delta = delta + np.add.outer(y * np.tan(y_tilt_rad), -x * np.tan(x_tilt_rad))

    return delta


def calculate_phase_profile(delta: np.ndarray, wavelength: float) -> np.ndarray:
    """Calculate the phase profile of the wave"""
    phase = 2 * np.pi * delta / wavelength  # % (2 * np.pi)

    return phase


def calculate_wavefront(
    phase: np.ndarray,
    passed_wavefront: np.ndarray,
    A: float,
    aperture: np.ndarray = None,
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
        # TODO ^ can remove this now that apertures work properly?

    # mask out apertured area
    if aperture is not None:
        wavefront[aperture] = 0 + 0j

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


def save_result_plots(
    result: SimulationResult,
    stage: SimulationStage,
    parameters: SimulationParameters,
    save_path: Path,
):
    """Plot and save the simulation results

    Args:
        result (SimulationResult): _description_
        stage (SimulationStage): _description_
        parameters (SimulationParameters): _description_
        save_path (Path): _description_
    """

    # save top-down
    fig = utils.plot_simulation(
        arr=result.top_down,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.start_distance,
        finish_distance=stage.finish_distance,
    )

    utils.save_figure(fig, os.path.join(save_path, "topdown.png"))
    plt.close(fig)

    fig = utils.plot_simulation(
        np.log(result.top_down + 10e-12),
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.start_distance,
        finish_distance=stage.finish_distance,
    )

    utils.save_figure(fig, os.path.join(save_path, "log_topdown.png"))
    plt.close(fig)

    fig = utils.plot_simulation(
        arr=result.side_on,
        pixel_size_x=parameters.pixel_size,
        start_distance=stage.start_distance,
        finish_distance=stage.finish_distance,
    )
    utils.save_figure(fig, os.path.join(save_path, "sideon.png"))
    plt.close(fig)

    if result.freq_arr is not None:
        fig = utils.plot_image(
            result.freq_arr,
            "Frequency Array",
            save=True,
            fname=os.path.join(save_path, "freq.png"),
        )
        plt.close(fig)

    if result.delta is not None:
        fig = utils.plot_image(
            result.delta,
            "Delta Profile",
            save=True,
            fname=os.path.join(save_path, "delta.png"),
        )
        plt.close(fig)

    if result.phase is not None:
        utils.plot_image(
            result.phase,
            "Phase Profile",
            save=True,
            fname=os.path.join(save_path, "phase.png"),
        )
        plt.close(fig)

    if result.lens is not None:
        fig = utils.plot_lens_profile_2D(result.lens)
        utils.save_figure(fig, fname=os.path.join(save_path, "lens_profile.png"))

        fig = utils.plot_lens_profile_slices(result.lens)
        utils.save_figure(fig, fname=os.path.join(save_path, "lens_slices.png"))

def invert_lens_and_output_medium(stage: SimulationStage, previous_stage: SimulationStage, parameters: SimulationParameters) -> SimulationStage:
    """Invert the lens profile, and swap the stage and lens mediums to create an 'inverse' lens

    Args:
        stage (SimulationStage): current simulation stage
        previous_stage (SimulationStage): previous simulation stage
        parameters (SimulationParameters): simulation parameters

    Raises:
        ValueError: simulation stage lens and medium are the same, lens has no effect.

    Returns:
        SimulationStage: update simulation stage, with 'inverse' lens
    """



    if (stage.lens.medium.refractive_index == previous_stage.output.refractive_index):
        raise ValueError("Lens and Medium on either side are the same Medium, Lens has no effect.")  # TODO: might be useful for someone...

    # change to 'air' lens, and invert the profile
    stage.lens = Lens(
        diameter=stage.lens.diameter,
        height=stage.lens.height,
        exponent=stage.lens.exponent,
        medium=previous_stage.output,
    )  # replace the lens with lens of previous output medium

    stage.lens.generate_profile(
        parameters.pixel_size, lens_type=parameters.lens_type
    )
    stage.lens.invert_profile()
    stage.lens_inverted = True

    return stage