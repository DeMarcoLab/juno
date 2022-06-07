import numpy as np
import pytest
from lens_simulation import Simulation
from lens_simulation.Lens import Lens, LensType
from lens_simulation.Medium import Medium
from lens_simulation.structures import SimulationParameters
from lens_simulation import utils


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
        lens_type=LensType.Spherical
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
    sim_profile = Simulation.pad_simulation(lens, sim_parameters)

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
    sim_profile = Simulation.pad_simulation(lens, sim_parameters)
    sim_n_pixels_height = utils._calculate_num_of_pixels(sim_parameters.sim_height, sim_parameters.pixel_size) 
    sim_n_pixels_width = utils._calculate_num_of_pixels(sim_parameters.sim_width, sim_parameters.pixel_size)
    assert sim_parameters.sim_height == sim_parameters.sim_width
    assert sim_profile.shape == (sim_n_pixels_height, sim_n_pixels_width)
    assert sim_profile[0, 0] == 0, "Corners should be zero"
    assert sim_profile[0, -1] == 0, "Corners should be zero"
    assert sim_profile[-1, 0] == 0, "Corners should be zero"
    assert sim_profile[-1, -1] == 0, "Corners should be zero"

