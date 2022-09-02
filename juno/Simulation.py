import os
import uuid
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import petname
import zarr
from scipy import fftpack
from tqdm import tqdm

from juno import plotting, utils, validation
from juno.beam import generate_beam
from juno.Lens import Lens, LensType, generate_lens
from juno.Medium import Medium
from juno.structures import (SimulationOptions,
                                        SimulationParameters, SimulationResult,
                                        SimulationStage, StageSettings)


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
        self.options = generate_simulation_options(config, log_dir)

        # common sim parameters
        self.parameters = generate_simulation_parameters(config)

        return config

    def setup_simulation(self, config: dict):

        # replace the existing config with a custom one if exists.
        for i, lens_config in enumerate(config["lenses"]):
            
            if lens_config["custom_config"] is not None:
                lens_config = utils.load_yaml_config(lens_config["custom_config"])
                lens_config = validation._validate_default_lens_config(lens_config)
                config["lenses"][i] = lens_config

        # generate all lenses for the simulations
        simulation_lenses = generate_lenses(config["lenses"], self.parameters)

        # generate all simulation stages
        self.sim_stages = generate_simulation_stages(config, simulation_lenses, self.parameters)

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

        # TODO: extract from the class
        progress_bar = tqdm(sim_stages, leave=False)
        for stage in progress_bar:

            progress_bar.set_description(
                f"Sim: {petname} ({str(sim_id)[-10:]}) - Propagating Wavefront"
            )


            if stage.wavefront is not None:
                propagation = stage.wavefront

            previous_wavefront = propagation

            # calculate stage phase profile
            phase = calculate_stage_phase(stage, parameters)

            # electric field (wavefront)
            amplitude: float = parameters.A if stage._id == 0 else 1.0
            wavefront = calculate_wavefront_v2(
                phase=phase,
                previous_wavefront=previous_wavefront,
                A=amplitude,
                aperture=stage.lens.aperture,
            ) 

            ## propagate wavefront #TODO: replace with v3 (vectorised)
            result = propagate_wavefront_v2(wavefront=wavefront, 
                                stage=stage, 
                                parameters=parameters, 
                                options=options)
            
            # pass the wavefront to the next stage
            propagation = result.propagation

            # save path
            save_path = os.path.join(options.log_dir, str(stage._id))

            if options.save_plot:

                # additional plotting items
                result.phase = phase

                progress_bar.set_description(
                    f"Sim: {petname} ({str(sim_id)[-10:]}) - Plotting Simulation"
                )

                plotting.save_result_plots(result, stage, parameters, save_path)

        # save final sim configruation
        config["finished"] = utils.current_timestamp()
        utils.save_metadata(config, options.log_dir)


def generate_simulation_options(config: dict, log_dir: str) -> SimulationOptions:

    options = SimulationOptions(
            log_dir=log_dir,
            save_plot=config["options"]["save_plot"],
            debug=config["options"]["debug"],
        )
    return options


def generate_simulation_parameters(config: dict) -> SimulationParameters:
    parameters = SimulationParameters(
        A=config["sim_parameters"]["A"],
        pixel_size=config["sim_parameters"]["pixel_size"],
        sim_width=config["sim_parameters"]["sim_width"],
        sim_height=config["sim_parameters"]["sim_height"],
        sim_wavelength=config["sim_parameters"]["sim_wavelength"],
    )

    return parameters


def generate_simulation_stages(config: dict, simulation_lenses: dict, parameters: SimulationParameters) -> list:
    """Generate the list of simulation stages

    Args:
        stages (list): config list containing simulation stages
        simulation_lenses (dict): config dict containing simulation lenses
        config (dict): simulation config
        parameters (SimulationParameters): simulation parameters

    Returns:
        list[SimulationStage]: list of SimulationStages ready for simulation
    """

    # validate sim, lens and medium setup
    stages = validation._validate_simulation_stage_list(config["stages"], simulation_lenses)

    # stages to be simulated
    sim_stages = []

    ################################ BEAM STAGE ################################

    # first stage is a beam
    beam_stage = generate_beam_simulation_stage(config, parameters)

    sim_stages.append(beam_stage)

    ################################ LENS STAGES ################################

    for i, stage in enumerate(stages):

        sim_stage_no = len(sim_stages)

        sim_stage = generate_simulation_stage(stage, simulation_lenses, parameters, sim_stage_no)

        # NOTE: if the lens and the output have the same medium, the lens is assumed to be 'double-sided'
        # therefore, we invert the lens profile to create an 'air lens' to properly simulate the double sided lens
        if (sim_stage.lens.medium.refractive_index == sim_stage.output.refractive_index):

            sim_stage = invert_lens_and_output_medium(sim_stage, sim_stages[sim_stage_no - 1], parameters)

        # update config
        config["stages"][i]["n_steps"] = len(sim_stage.distances)
        config["stages"][i]["start_distance"] = sim_stage.distances[0]
        config["stages"][i]["finish_distance"] = sim_stage.distances[-1]
        config["stages"][i]["lens_inverted"] = sim_stage.lens_inverted

        # add to simulation
        sim_stages.append(sim_stage)

    return sim_stages


def load_sim_stage_config(sim_config):

    stage_settings = StageSettings(
        lens = sim_config["lens"],
        output = sim_config["output"],
        n_steps = sim_config["n_steps"],
        step_size = sim_config["step_size"],
        start_distance = sim_config["start_distance"],
        finish_distance = sim_config["finish_distance"],
        use_equivalent_focal_distance = bool(sim_config["use_equivalent_focal_distance"]),
        focal_distance_start_multiple = sim_config["focal_distance_start_multiple"],
        focal_distance_multiple = sim_config["focal_distance_multiple"],
    )

    return stage_settings



def generate_simulation_stage(stage_config: dict, simulation_lenses: dict, parameters: SimulationParameters, sim_stage_no: int = 0) -> SimulationStage:

    # TODO: replace
    settings = load_sim_stage_config(stage_config)

    stage = SimulationStage(
        lens=simulation_lenses.get(stage_config.get("lens")),
        output=Medium(stage_config.get("output"), parameters.sim_wavelength),
        distances=None,
        lens_inverted=False,
        _id=sim_stage_no,
    )


    # dynamically calculate distance based on focal distance
    if stage_config["use_equivalent_focal_distance"] is True:
        start_distance, finish_distance = calculate_start_and_finish_distance_v2(stage.lens, stage.output,
                    stage_config.get("focal_distance_start_multiple"),
                    stage_config.get("focal_distance_multiple"))
    else:
        start_distance = stage_config.get("start_distance")
        finish_distance = stage_config.get("finish_distance")

    # calculate propagation distances
    stage.distances = calculate_propagation_distances(
                                    start_distance=start_distance,
                                    finish_distance=finish_distance,
                                    n_steps=stage_config.get("n_steps"),
                                    step_size=stage_config.get("step_size"))

    return stage


def generate_beam_simulation_stage(config: dict, parameters: SimulationParameters) -> SimulationStage:

    beam = generate_beam(config["beam"], parameters)
    beam_stage = SimulationStage(
        lens=beam.lens,
        output=beam.output_medium,
        tilt={"x": beam.tilt[0], "y": beam.tilt[1]},
    )

    # calculate propagation distances
    beam_stage.distances = calculate_propagation_distances(
                                    start_distance=beam.start_distance,
                                    finish_distance=beam.finish_distance,
                                    n_steps=beam.settings.n_steps,
                                    step_size=beam.settings.step_size)

    # create beam initial wavefront
    beam_stage.wavefront = beam.wavefront

    return beam_stage


def calculate_start_and_finish_distance_v2(lens: Lens, output: Medium, start_multiple: float, finish_multiple: float):

    eq_fd = calculate_equivalent_focal_distance(lens, output)

    start_distance = (start_multiple * eq_fd)
    finish_distance = (finish_multiple * eq_fd)

    return start_distance, finish_distance


def calculate_propagation_distances(start_distance: float, finish_distance: float, n_steps: int, step_size: float = None) -> np.ndarray:

    if start_distance >= finish_distance:
        raise ValueError(f"start_distance ({start_distance}) is greater than finish_distance ({finish_distance}).")

    # calculate n_steps
    n_steps = calculate_num_steps_in_distance(start_distance, finish_distance, step_size, n_steps)

    # calculate propagation distances
    distances = np.linspace(start_distance, finish_distance, n_steps)

    return distances


def calculate_num_steps_in_distance(start, finish, step_size, n_steps) -> int:

    if n_steps in [0, None] and step_size in [0, None]:
        raise ValueError(f"Both n_steps ({n_steps}) and step_size ({step_size}) cannot be zero or None.")

    if step_size not in [0, None]:
        if step_size > (finish - start):
            raise ValueError(f"The step_size ({step_size}) is greater than the distance range: {finish - start}m.")

        n_steps = int(np.ceil((finish + 1e-12 - start) / step_size))

    if n_steps in [0, None]:
        raise ValueError(f"The number of steps is zero, this will cause an error in propagation. Please increase n_steps, or step_size. n_steps {n_steps}, step_size: {step_size}")

    return n_steps

def generate_lenses(lenses: list, parameters: SimulationParameters):
    """Generate all the lenses for the simulation"""

    simulation_lenses = {}
    for lens_config in lenses:

        medium = Medium(refractive_index=lens_config["medium"],
                wavelength=parameters.sim_wavelength)

        # generate lens from config
        lens = generate_lens(lens_config=lens_config,
                        medium=medium,
                        pixel_size=parameters.pixel_size)

        if lens.lens_type is LensType.Spherical:
            # check lens fits in the simulation
            if lens.diameter > parameters.sim_width or lens.diameter > parameters.sim_height:
                raise ValueError(
                    f"Lens diameter must be smaller than the simulation size: lens: {lens.diameter:.2e}m, sim: {parameters.sim_width:.2e}mx{parameters.sim_height:.2e}m"
                )

        if lens.lens_type is LensType.Cylindrical:
            if lens.diameter > parameters.sim_width or lens.length > parameters.sim_height:
                raise ValueError(
                    f"Lens must be smaller than the simulation size: lens: {lens.diameter:.2e}mx{lens.length:.2e}m, sim: {parameters.sim_width:.2e}mx{parameters.sim_height:.2e}m"
                )


        # check the escape path fits within then simulation
        if lens_config["escape_path"] is not None:
            test_escape_path_fits_inside_simulation(lens, parameters, lens_config["escape_path"])

        # pad the lens profile to be the same size as the simulation
        lens = pad_simulation(lens, parameters=parameters)

        # apply all aperture masks
        lens.apply_aperture_masks()

        simulation_lenses[lens_config.get("name")] = lens

    return simulation_lenses


def propagate_wavefront_v3(
    wavefront: np.ndarray,
    stage: SimulationStage,
    parameters: SimulationParameters,
    options: SimulationOptions,
) -> SimulationResult:
    """Propagate the light wavefront using the supplied settings and parameters. Vectorised version.

    Args:
        wavefront (np.ndarray): the initial wavefront to propagate from.
        sim_stage (SimulationStage): the setup of the simulation stage, lens -> output
        parameters (SimulationParameters): the global simulation parameters (shared for all stages)
        options (SimulationOptions): global simulation options

    Returns:
        SimulationResult: results of the wave propagation (including intermediates if debugging)
    """
    output_medium: Medium = stage.output
    distances: np.ndarray = stage.distances

    save_path = os.path.join(options.log_dir, str(stage._id))

    # fourier transform of wavefront
    fft_wavefront = fftpack.fft2(wavefront)
    
    # generate frequency array
    freq_arr = generate_sq_freq_arr(wavefront, pixel_size=parameters.pixel_size)

    # pre-allocate sim
    fname = os.path.join(save_path, f"sim.zarr")
    sim_shape = (distances.shape[0], wavefront.shape[0], wavefront.shape[1])
    sim = zarr.open(fname, mode="w",
                    shape=sim_shape,
                    dtype=np.float32)

    wave_number = output_medium.wave_number

    # vector part, broadcast
    # (10,)  -> (10, 251, 251)
    d_vec = np.broadcast_to(distances[:, np.newaxis, np.newaxis], (distances.shape[0], *freq_arr.shape))

    prop1 = np.exp(1j * wave_number * d_vec)
    prop2 = np.exp((-1j * 2 * np.pi ** 2 * d_vec * freq_arr ) / wave_number)
    prop = prop1 * prop2

    propagation = fftpack.ifft2(prop * fft_wavefront)

    output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2) ** 2

    sim[:] = np.round(output.astype(np.float32), 10)
    propagation = propagation[-1, :, :]

    # return results
    result = SimulationResult(
        propagation=propagation,
        sim=sim,
        lens=stage.lens,
        freq_arr=freq_arr,
        delta=None,
        phase=None,
    )

    return result


def propagate_wavefront_v2(
    wavefront: np.ndarray,
    stage: SimulationStage,
    parameters: SimulationParameters,
    options: SimulationOptions,
) -> SimulationResult:
    """Propagate the light wavefront using the supplied settings and parameters.

    Args:
        wavefront (np.ndarray): the initial wavefront to propagate from.
        sim_stage (SimulationStage): the setup of the simulation stage, lens -> output
        parameters (SimulationParameters): the global simulation parameters (shared for all stages)
        options (SimulationOptions): global simulation options

    Returns:
        SimulationResult: results of the wave propagation (including intermediates if debugging)
    """
    output_medium: Medium = stage.output
    distances: np.ndarray = stage.distances

    save_path = os.path.join(options.log_dir, str(stage._id))

    # fourier transform of wavefront
    fft_wavefront = fftpack.fft2(wavefront)
    
    # generate frequency array
    freq_arr = generate_sq_freq_arr(wavefront, pixel_size=parameters.pixel_size)

    # pre-allocate sim
    fname = os.path.join(save_path, f"sim.zarr")
    sim_shape = (len(distances), wavefront.shape[0], wavefront.shape[1])
    sim = zarr.open(fname, mode="w",
                    shape=sim_shape,
                    # chunks=(1000, 1000),  # note dont manaully set chunk size
                    dtype=np.float32)

    # propagate the wavefront over distance
    prop_progress_bar = tqdm(distances, leave=False)
    for i, distance in enumerate(prop_progress_bar):
        prop_progress_bar.set_description(
            f"Propagating Wavefront at Distance {distance:.4f} / {distances[-1]:.4f}m"
        )

        rounded_output, propagation = propagate_over_distance(
            fft_wavefront, distance, freq_arr, output_medium.wave_number
        )

        sim[i, :, :] = rounded_output # NOTE: rounded output must fit in RAM, next target

    # return results
    result = SimulationResult(
        propagation=propagation,
        sim=sim,
        lens=stage.lens,
        freq_arr=freq_arr,
        delta=None,
        phase=None,
    )

    return result

def calculate_stage_phase(stage: SimulationStage, parameters: SimulationParameters) -> np.ndarray:
    """Calculate the phase profile for the simulation stage."""

    # delta (optical path distance)
    delta = calculate_tilted_delta_profile(stage.lens.profile, stage.lens, stage.output, stage.tilt)

    # phase
    phase = calculate_phase_profile(delta=delta, wavelength=parameters.sim_wavelength)

    return phase

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


def pad_simulation(lens: Lens, parameters: SimulationParameters) -> Lens:
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

    # TODO: fix better
    # need to check this for 1d case
    if y0 == 0 and y1 == 0:
        y1 += 1
    if x0 == 0 and x1 == 0:
        x1 +=1

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
        x = np.arange(sim_profile.shape[1]) * lens.pixel_size
        y = np.arange(sim_profile.shape[0]) * lens.pixel_size

        y_tilt_rad = np.deg2rad(tilt["y"])
        x_tilt_rad = np.deg2rad(tilt["x"])

        # modify the optical path of the light based on tilt
        delta = delta + np.add.outer(y * np.tan(y_tilt_rad), x * np.tan(x_tilt_rad))

    return delta


def calculate_phase_profile(delta: np.ndarray, wavelength: float) -> np.ndarray:
    """Calculate the phase profile of the wave"""
    phase = 2 * np.pi * delta / wavelength  # % (2 * np.pi)

    return phase

def calculate_wavefront_v2(
    phase: np.ndarray,
    previous_wavefront: np.ndarray,
    A: float,
    aperture: np.ndarray = None,
) -> np.ndarray:
    """Calculate the wavefront of light. (Electric Field)"""

    # only amplifiy the first stage propagation
    wavefront = A * np.exp(1j * phase) * previous_wavefront

    # padded area should be 0+0j
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
        raise ValueError("Lens and Medium on either side are the same Medium, Lens has no effect.")

    # change to 'air' lens, and invert the profile
    stage.lens = Lens(
        diameter=stage.lens.diameter,
        height=stage.lens.height,
        exponent=stage.lens.exponent,
        medium=previous_stage.output,
    )  # replace the lens with lens of previous output medium

    stage.lens.generate_profile(parameters.pixel_size)
    stage.lens.invert_profile()
    stage.lens_inverted = True

    return stage



def test_escape_path_fits_inside_simulation(lens: Lens, parameters, ep: float):
    from juno import utils
    from juno.Lens import calculate_escape_path_dimensions
    from juno.structures import SimulationParameters

    n_pixels_sim_height = utils._calculate_num_of_pixels(
        parameters.sim_height, parameters.pixel_size
    )
    n_pixels_sim_width = utils._calculate_num_of_pixels(
        parameters.sim_width, parameters.pixel_size
    )

    # calculate the escape path dimensions
    ep_h, ep_w = calculate_escape_path_dimensions(lens, ep)

    if ep_h > n_pixels_sim_height:
        raise ValueError(
            f"The given escape path is outside the simulation size: ep: {ep_h}px, sim: {n_pixels_sim_height}px"
        )

    if ep_w > n_pixels_sim_width:
        raise ValueError(
            f"The given escape path is outside the simulation size: ep: {ep_w}px, sim: {n_pixels_sim_width}px"
        )
