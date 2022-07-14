from matplotlib.pyplot import step
import numpy as np
import pytest
from star_glass import Simulation
from star_glass.Lens import Lens, LensType, generate_lens
from star_glass.Medium import Medium
from star_glass.structures import SimulationParameters, SimulationStage
from star_glass import utils


LENS_DIAMETER = 100e-6
LENS_HEIGHT = 20e-6
LENS_FOCUS_EXPONENT = 2.0
LENS_AXICON_EXPONENT = 1.0
LENS_PIXEL_SIZE = 1e-6

@pytest.fixture
def spherical_lens():
    # create lens
    lens = Lens(diameter=LENS_DIAMETER,
                height=LENS_HEIGHT,
                exponent=LENS_FOCUS_EXPONENT,
                medium=Medium(2.348))

    lens.generate_profile(LENS_PIXEL_SIZE, lens_type=LensType.Spherical)

    return lens

@pytest.fixture
def cylindrical_lens():
    # create lens
    lens = Lens(diameter=LENS_DIAMETER,
                height=LENS_HEIGHT,
                exponent=LENS_FOCUS_EXPONENT,
                medium=Medium(2.348))

    lens.generate_profile(LENS_PIXEL_SIZE, lens_type=LensType.Cylindrical)

    return lens

@pytest.fixture
def sim_parameters():
    return SimulationParameters(
        A=10000,
        pixel_size=200.e-9,
        sim_width=LENS_DIAMETER,
        sim_height=LENS_DIAMETER,
        sim_wavelength=488.e-9,
    )


@pytest.mark.parametrize(
    "pixel_size, expected",
    [(1, [0.0, 0.0625, 0.25, 0.0625]), (2, [0.0, 0.015625, 0.0625, 0.015625])],
)
def test_generate_squared_frequency_array_even(pixel_size, expected):
    array = np.array(np.ones(4))
    n_pixels = len(array)
    frequency_array_new = Simulation.generate_squared_frequency_array(
        n_pixels, pixel_size
    )

    assert len(frequency_array_new) == len(array)
    assert np.allclose(frequency_array_new, expected)


@pytest.mark.parametrize(
    "pixel_size, expected",
    [(1, [0.0, 0.04, 0.16, 0.16, 0.04]), (2, [0.0, 0.01, 0.04, 0.04, 0.01])],
)
def test_generate_squared_frequency_array_odd(pixel_size, expected):
    array = np.array(np.ones(5))
    n_pixels = len(array)
    frequency_array_new = Simulation.generate_squared_frequency_array(
        n_pixels, pixel_size
    )

    assert len(frequency_array_new) == len(array)
    assert np.allclose(frequency_array_new, expected)


# TODO: test_gen_sq_freq_arr
# for 1d, and 2d cases, and error


def test_calculate_equivalent_focal_distance_large():
    medium = Medium(1.0)
    lens = Lens(200, 20, 2.0, Medium(1.5))

    focal_distance = Simulation.calculate_equivalent_focal_distance(lens, medium)
    assert np.isclose(focal_distance, 520, rtol=1e-6)


@pytest.mark.parametrize("lens_exponent", [1.0, 1.5, 2.0, 2.1])
def test_calculate_equivalent_focal_distance(lens_exponent):
    # all exponents should result in equivalent focal distance
    medium = Medium(1.0)
    lens = Lens(4500e-6, 70e-6, lens_exponent, Medium(2.348))

    focal_distance = Simulation.calculate_equivalent_focal_distance(lens, medium)
    assert np.isclose(focal_distance, 0.0268514, rtol=1e-6)


@pytest.mark.parametrize("lens_exponent", [1.0, 1.5, 2.0, 2.1])
def test_calculate_equivalent_focal_distance_fail_due_to_height(lens_exponent):
    # changing height changes equivalent focal distance for all exponents
    medium = Medium(1.0)
    lens = Lens(4500e-6, 80e-6, lens_exponent, Medium(2.348))

    focal_distance = Simulation.calculate_equivalent_focal_distance(lens, medium)
    assert not np.isclose(focal_distance, 0.0268514, rtol=1e-6)



def test_calculate_number_of_pixels(sim_parameters):

    sim_width = sim_parameters.sim_width
    pixel_size = sim_parameters.pixel_size

    # odd
    sim_n_pixels = utils._calculate_num_of_pixels(sim_width, pixel_size, odd=True)

    n_pixels = sim_width // pixel_size
    if n_pixels % 2 == 0:
        n_pixels += 1

    assert sim_n_pixels == n_pixels

    # even
    sim_n_pixels_even = utils._calculate_num_of_pixels(sim_width, pixel_size, odd=False)

    assert sim_n_pixels_even == sim_width // pixel_size


def test_pad_simulation_asymmetric(sim_parameters):
    """Only pad along the second axis for asymmetric simulation"""
    
    # asymmetric sim height
    sim_parameters.sim_height = LENS_DIAMETER * 0.75

    lens = Lens(diameter=LENS_DIAMETER / 2, 
        height=20e-6, 
        exponent=2.0, 
        medium=Medium(1))

    lens.generate_profile(sim_parameters.pixel_size, LensType.Cylindrical)
    sim_lens = Simulation.pad_simulation(lens, sim_parameters)
    sim_profile = sim_lens.profile

    sim_n_pixels_height = utils._calculate_num_of_pixels(sim_parameters.sim_height, sim_parameters.pixel_size) 
    sim_n_pixels_width = utils._calculate_num_of_pixels(sim_parameters.sim_width, sim_parameters.pixel_size) 
    assert sim_parameters.sim_height != sim_parameters.sim_width
    assert sim_profile.shape == (sim_n_pixels_height, sim_n_pixels_width)
    assert sim_profile[0, 0] == 0, "Corners should be zero"
    assert sim_profile[0, -1] == 0, "Corners should be zero"
    assert sim_profile[-1, 0] == 0, "Corners should be zero"
    assert sim_profile[-1, -1] == 0, "Corners should be zero"

def test_pad_simulation_symmetric(sim_parameters):
    
    lens = Lens(diameter=LENS_DIAMETER / 2, 
    height=20e-6, 
    exponent=2.0, 
    medium=Medium(1))
    
    lens.generate_profile(sim_parameters.pixel_size, LensType.Spherical)
    sim_lens = Simulation.pad_simulation(lens, sim_parameters)
    sim_profile = sim_lens.profile
    sim_n_pixels_height = utils._calculate_num_of_pixels(sim_parameters.sim_height, sim_parameters.pixel_size) 
    sim_n_pixels_width = utils._calculate_num_of_pixels(sim_parameters.sim_width, sim_parameters.pixel_size)
    
    assert sim_parameters.sim_height == sim_parameters.sim_width
    assert sim_profile.shape == (sim_n_pixels_height, sim_n_pixels_width)
    assert sim_profile[0, 0] == 0, "Corners should be zero"
    assert sim_profile[0, -1] == 0, "Corners should be zero"
    assert sim_profile[-1, 0] == 0, "Corners should be zero"
    assert sim_profile[-1, -1] == 0, "Corners should be zero"


## SIMULATION SETUP
import os

@pytest.fixture
def config_with_sweep():

    config = utils.load_config(
        os.path.join(os.path.dirname(__file__), "test_config_with_sweep.yaml")
    )

    return config


def test_Simulation_init():

    return NotImplemented

## GENERATE SIM PROPERTIES AND DATA

def test_generate_simulation_parameters(config_with_sweep):
    config = config_with_sweep

    parameters = Simulation.generate_simulation_parameters(config)

    assert parameters.A == config["sim_parameters"]["A"]
    assert parameters.pixel_size == config["sim_parameters"]["pixel_size"]
    assert parameters.sim_width == config["sim_parameters"]["sim_width"]
    assert parameters.sim_height == config["sim_parameters"]["sim_height"]
    assert parameters.sim_wavelength == config["sim_parameters"]["sim_wavelength"]


def test_generate_simulation_options(config_with_sweep):

    config = config_with_sweep

    options = Simulation.generate_simulation_options(config, "log")

    assert options.log_dir == "log"
    assert options.save_plot == config["options"]["save_plot"]
    assert options.debug == config["options"]["debug"]



def test_generate_lenses(config_with_sweep):
    config = config_with_sweep

    parameters = Simulation.generate_simulation_parameters(config)

    simulation_lenses = Simulation.generate_lenses(config["lenses"], parameters)

    assert len(simulation_lenses) == len(config.get("lenses")), f"The number of generated lenses is different than specified in the config. generated: {len(simulation_lenses)} , config: {len(config.get('lenses'))} "

    for lc in config.get("lenses"):
        
        lens = simulation_lenses.get(lc["name"])  # get lens by name, check parameters are the same
        assert lens.diameter == lc.get("diameter")
        assert lens.height == lc.get("height")
        assert lens.exponent == lc.get("exponent")
        assert lens.pixel_size == parameters.pixel_size

def test_generate_lens_raises_error_for_lens_larger_than_sim(config_with_sweep):

    config = config_with_sweep
    config["sim_parameters"]["sim_width"] = 1.e-6
    config["sim_parameters"]["sim_height"] = 1.e-6

    parameters = Simulation.generate_simulation_parameters(config)

    # raises value error if lens larger than sim width / height
    with pytest.raises(ValueError):
        simulation_lenses = Simulation.generate_lenses(config["lenses"], parameters)
    

def test_generate_lens_raises_error_for_lens_escape_path_outside_sim(config_with_sweep):

    config = config_with_sweep
    config["lenses"][0]["escape_path"] = 2.0

    parameters = Simulation.generate_simulation_parameters(config)

    # raises value error because escape path outside sim
    with pytest.raises(ValueError):
        simulation_lenses = Simulation.generate_lenses(config["lenses"], parameters)
    


def test_generate_simulation_stages(config_with_sweep):

    config = config_with_sweep

    parameters = Simulation.generate_simulation_parameters(config)
    simulation_lenses = Simulation.generate_lenses(config["lenses"], parameters)

    simulation_stages = Simulation.generate_simulation_stages(config, simulation_lenses, parameters)

    # needs to be +1 to account for beam stage
    assert len(simulation_stages) == len(config["stages"]) + 1,   f"The number of generated stages is different than specified in the config. generated: {len(simulation_stages)} , config: {len(config.get('stages'))} "

    for ss, sc in zip(simulation_stages, config["stages"]):

        # check parameters are assigned correctly
        assert ss.output.refractive_index == sc.get("output")
        assert ss.output.wavelength == parameters.sim_wavelength


def test_generate_simulation_stage(config_with_sweep):

    config = config_with_sweep

    parameters = Simulation.generate_simulation_parameters(config)
    simulation_lenses = Simulation.generate_lenses(config["lenses"], parameters)

    sim_config = config["stages"][0]
    sim_config["use_equivalent_focal_distance"] = False
    sim_config["step_size"] = None
    stage = Simulation.generate_simulation_stage(sim_config, simulation_lenses, parameters, 0)

    assert stage.output.refractive_index == sim_config.get("output")
    assert stage.output.wavelength == parameters.sim_wavelength
    assert len(stage.distances) == sim_config.get("n_steps")
    assert stage.distances[0] == sim_config.get("start_distance")
    assert stage.distances[-1] == sim_config.get("finish_distance")
    assert stage._id == 0

def test_generate_simulation_stage_with_dynamic_n_steps(config_with_sweep):

    config = config_with_sweep

    parameters = Simulation.generate_simulation_parameters(config)
    simulation_lenses = Simulation.generate_lenses(config["lenses"], parameters)

    sim_config = config["stages"][0]
    sim_config["use_equivalent_focal_distance"] = False
    stage = Simulation.generate_simulation_stage(sim_config, simulation_lenses, parameters, 0)
    
    n_steps = Simulation.calculate_num_steps_in_distance(
                    sim_config.get("start_distance"), 
                    sim_config.get("finish_distance"), 
                    sim_config.get("step_size"), 
                    None)

    assert len(stage.distances) == n_steps
    assert stage.distances[0] == sim_config.get("start_distance")
    assert stage.distances[-1] == sim_config.get("finish_distance")


def test_create_beam_simulation_stage(config_with_sweep):
    config = config_with_sweep

    config["beam"]["step_size"] = None
    parameters = Simulation.generate_simulation_parameters(config)
    beam_stage = Simulation.generate_beam_simulation_stage(config, parameters)

    assert beam_stage.output.refractive_index == config["beam"]["output_medium"]
    assert len(beam_stage.distances) == config["beam"]["n_steps"]
    assert beam_stage.distances[0] == 0
    assert beam_stage.distances[-1] == config["beam"]["source_distance"]
    assert beam_stage.tilt["x"] == config["beam"]["tilt_x"]
    assert beam_stage.tilt["y"] == config["beam"]["tilt_y"]

    # with dynamic n_steps
    config["beam"]["step_size"] = (config["beam"]["source_distance"])  / 10
    beam_stage = Simulation.generate_beam_simulation_stage(config, parameters)

    assert len(beam_stage.distances) == 10 + 1



def test_calculate_num_steps_in_distance():

    start_distance = 0.0
    finish_distance = 10.0
    step_size = 1.0
    n_steps = 0.0

    n_steps = Simulation.calculate_num_steps_in_distance(start_distance, finish_distance, step_size, n_steps)

    assert n_steps == 11

def test_calculate_num_steps_in_distance_raises_error():

    # zero step size, zero nsteps
    start_distance, finish_distance, step_size, n_steps = 0.0, 10.0, 0.0, 0.0
    with pytest.raises(ValueError):
        n_steps = Simulation.calculate_num_steps_in_distance(start_distance, finish_distance, step_size, n_steps)
    
    # step size greater than distance
    start_distance, finish_distance, step_size, n_steps = 0.0, 10.0, 20.0, 0.0
    with pytest.raises(ValueError):
        n_steps = Simulation.calculate_num_steps_in_distance(start_distance, finish_distance, step_size, n_steps)
    


def test_calculate_start_and_finish_distance(config_with_sweep):

    config = config_with_sweep

    # TODO:
    # parameters = Simulation.generate_simulation_parameters(config)
    # simulation_lenses = Simulation.generate_lenses(config["lenses"], parameters)

    # sim_config = config["stages"][0]
    # lens = simulation_lenses.get(sim_config["lens"])
    # stage = Simulation.generate_simulation_stage(sim_config, lens, parameters, 0)



def test_calculate_propagation_distances():

    sd, fd, step_size, n_steps = 0, 10.e-3, 0, 3
    distances = Simulation.calculate_propagation_distances(sd, fd, n_steps, step_size) 
    assert np.array_equal(distances, [0, 5.e-3, 10.e-3]), "Distance arrays should be equal"
    
    sd, fd, step_size, n_steps = 0, 10.e-3, 5.e-3, 0
    distances = Simulation.calculate_propagation_distances(sd, fd, n_steps, step_size) 
    assert np.array_equal(distances, [0, 5.e-3, 10.e-3]), "Distance arrays should be equal"


def test_calculate_propagation_distances_raises_error():

    # start greater than finish
    sd, fd, step_size, n_steps = 10, 0, 0, 3
    with pytest.raises(ValueError):
        distances = Simulation.calculate_propagation_distances(sd, fd, n_steps, step_size) 
    
    # step size and n_steps are both zero
    sd, fd, step_size, n_steps = 0, 10.e-3, 0, 0
    with pytest.raises(ValueError):
        distances = Simulation.calculate_propagation_distances(sd, fd, n_steps, step_size) 
    
    # step size larger than range
    sd, fd, step_size, n_steps = 0, 10, 20, 1
    with pytest.raises(ValueError):
        distances = Simulation.calculate_propagation_distances(sd, fd, n_steps, step_size) 
    
    # n_steps is zero, with no step_size defined
    sd, fd, step_size, n_steps = 0, 10, 0, 0
    with pytest.raises(ValueError):
        distances = Simulation.calculate_propagation_distances(sd, fd, n_steps, step_size) 
    