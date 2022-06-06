import numpy as np
import pytest
from lens_simulation import Simulation, Lens
from lens_simulation.Lens import LensType
from lens_simulation.structures import SimulationParameters
from lens_simulation import utils

from tests.test_Lens import LENS_DIAMETER, spherical_lens



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


def test_calculate_equivalent_focal_distance_large():
    medium = Lens.Medium(1.0)
    lens = Lens.Lens(200, 20, 2.0, Lens.Medium(1.5))

    focal_distance = Simulation.calculate_equivalent_focal_distance(lens, medium)
    assert np.isclose(focal_distance, 520, rtol=1e-6)


@pytest.mark.parametrize("lens_exponent", [1.0, 1.5, 2.0, 2.1])
def test_calculate_equivalent_focal_distance(lens_exponent):
    # all exponents should result in equivalent focal distance
    medium = Lens.Medium(1.0)
    lens = Lens.Lens(4500e-6, 70e-6, lens_exponent, Lens.Medium(2.348))

    focal_distance = Simulation.calculate_equivalent_focal_distance(lens, medium)
    assert np.isclose(focal_distance, 0.0268514, rtol=1e-6)


@pytest.mark.parametrize("lens_exponent", [1.0, 1.5, 2.0, 2.1])
def test_calculate_equivalent_focal_distance_fail_due_to_height(lens_exponent):
    # changing height changes equivalent focal distance for all exponents
    medium = Lens.Medium(1.0)
    lens = Lens.Lens(4500e-6, 80e-6, lens_exponent, Lens.Medium(2.348))

    focal_distance = Simulation.calculate_equivalent_focal_distance(lens, medium)
    assert not np.isclose(focal_distance, 0.0268514, rtol=1e-6)

# def test_pad_simulation():

#     # create lens
#     lens = Lens.Lens(diameter=4500e-6, 
#                 height=20e-6, 
#                 exponent=2.0, 
#                 medium=Lens.Medium(1))

#     # default horizontal padding for extrude
#     lens.generate_profile(1e-6, LensType.Cylindrical)
#     pad_px = lens.profile.shape[-1]
#     sim_profile = Simulation.pad_simulation(lens, pad_px=pad_px)
#     assert sim_profile.shape ==  (1, lens.profile.shape[0] + 2 * pad_px)
#     assert np.allclose(sim_profile[:, :pad_px], 0)   # padded areas should be zero
#     assert np.allclose(sim_profile[:, -pad_px:], 0)  # padded areas should be zero


#     # symmetric padding for revolve
#     lens.generate_profile(1e-6, LensType.Spherical)
#     pad_px=lens.profile.shape[-1]
#     sim_profile = Simulation.pad_simulation(lens, pad_px=pad_px)
#     assert sim_profile.shape == (lens.profile.shape[0] + 2 * pad_px, lens.profile.shape[1] + 2*pad_px)
#     assert np.allclose(sim_profile[:lens.profile.shape[0], :], 0)   # padded areas should be zero
#     assert np.allclose(sim_profile[:, :lens.profile.shape[1]], 0)   # padded areas should be zero
#     assert np.allclose(sim_profile[-lens.profile.shape[0]:, :], 0)  # padded areas should be zero
#     assert np.allclose(sim_profile[:, -lens.profile.shape[1]:], 0)  # padded areas should be zero


@pytest.fixture
def sim_parameters():
    return SimulationParameters(
        A=10000,
        pixel_size=200.e-9,
        sim_width=LENS_DIAMETER,
        sim_wavelength=488.e-9,
        lens_type=LensType.Spherical
    )

def test_pad_simulation_lens_width_for_same_size(spherical_lens, sim_parameters):

    # lens = Lens.Lens(diameter=700.e-6, 
    #         height=20e-6, 
    #         exponent=2.0, 
    #         medium=Lens.Medium(1))

    lens = spherical_lens

    # check profile shape before / after
    # check for both lens types

    lens.generate_profile(sim_parameters.pixel_size, LensType.Cylindrical)
    pre_shape = lens.profile.shape
    sim_profile = Simulation.pad_simulation(lens, sim_parameters)
    assert sim_profile.shape == pre_shape
    
    lens.generate_profile(sim_parameters.pixel_size, LensType.Spherical)
    pre_shape = lens.profile.shape
    sim_profile = Simulation.pad_simulation(lens, sim_parameters)
    assert sim_profile.shape == pre_shape


def test_pad_simulation_asymmetric(sim_parameters):
   
    lens = Lens.Lens(diameter=LENS_DIAMETER / 2, 
        height=20e-6, 
        exponent=2.0, 
        medium=Lens.Medium(1))

    lens.generate_profile(sim_parameters.pixel_size, LensType.Cylindrical)
    print("lens_shape: ", lens.profile.shape)
    sim_profile = Simulation.pad_simulation(lens, sim_parameters)
    sim_n_pixels = utils._calculate_num_of_pixels(sim_parameters.sim_width, sim_parameters.pixel_size) 
    assert sim_profile.shape == (1, sim_n_pixels)

def test_pad_simulation_symmetric(sim_parameters):
    
    lens = Lens.Lens(diameter=LENS_DIAMETER / 2, 
    height=20e-6, 
    exponent=2.0, 
    medium=Lens.Medium(1))
    
    lens.generate_profile(sim_parameters.pixel_size, LensType.Spherical)
    sim_profile = Simulation.pad_simulation(lens, sim_parameters)
    sim_n_pixels = utils._calculate_num_of_pixels(sim_parameters.sim_width, sim_parameters.pixel_size) 
    assert sim_profile.shape == (sim_n_pixels, sim_n_pixels)



