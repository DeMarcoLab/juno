import numpy as np
import pytest
from lens_simulation import Simulation, Lens


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


def test_generate_discrete_profile():
    """with default rounding the lens is binarised to integer values"""
    pixel_size = 1e-3
    z_step = 1

    lens = Lens.Lens(diameter=5, height=3, exponent=0.5, medium=Lens.Medium(2.348))
    lens.generate_profile(pixel_size=pixel_size)

    discrete_profile = Simulation.generate_discrete_profile(
        lens=lens, z_resolution=z_step, output_medium=Lens.Medium(1.5),
    )

    assert np.ndim(discrete_profile) == np.ndim(lens.profile) + 1

    assert discrete_profile.shape[0] == 4

    # assert that all values are multiples of step size
    for layer in discrete_profile:
        for pixel in layer:
            assert pixel % z_step == 0
