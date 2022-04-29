import pytest

from lens_simulation.Lens import Lens, Medium, LensType

import numpy as np


def test_Lens():

    lens = Lens(diameter=1.0, height=1.0, exponent=2.0)

    assert lens.diameter == 1.0
    assert lens.height == 1.0
    assert lens.exponent == 2.0


def test_axicon_lens():

    lens = Lens(diameter=1.0, height=1.0, exponent=1.0)

    profile = lens.generate_profile(pixel_size=10e-9, lens_type=LensType.Cylindrical)

    assert np.isclose(np.round(np.min(profile), 7), 0)  # TODO: broken
    assert np.isclose(np.max(profile), lens.height)
    assert profile[int(len(profile) * 0.75)] == lens.height / 2


def test_focusing_lens():

    lens = Lens(diameter=1.0, height=1.0, exponent=2.0)

    profile = lens.generate_profile(pixel_size=10e-9, lens_type=LensType.Cylindrical)

    print(np.min(profile))
    print(np.max(profile))

    # TODO: check here on the tolerance values. i.e. FIB, print resolution

    assert np.isclose(0, np.min(profile), atol=100e-9)
    assert np.isclose(np.max(profile), lens.height, rtol=10e-6)
    assert not profile[int(len(profile) * 0.75)] == -lens.height / 2


# def test_extrude_lens():

#     lens = Lens(diameter=4500e-6, 
#                 height=20e-6, 
#                 exponent=2.0, 
#                 medium=Medium(1))
#     base_profile = lens.generate_profile(1e-6)

#     lens.extrude_profile(length=10e-6)

#     for profile_1D in lens.profile:

#         assert np.array_equal(profile_1D, base_profile), "Extruded profile is different than base profile."

def test_revolve_lens():

    lens = Lens(diameter=4500e-6, 
                height=20e-6, 
                exponent=2.0, 
                medium=Medium(1))
    profile_2D = lens.generate_profile(1e-6, lens_type=LensType.Spherical)

    
    # corners should be zero
    assert profile_2D[0, 0] == 0, "Corner point should be zero"
    assert profile_2D[0, profile_2D.shape[1]-1] == 0, "Corner point should be zero"
    assert profile_2D[profile_2D.shape[0]-1, 0] == 0, "Corner point should be zero"
    assert profile_2D[profile_2D.shape[0]-1, profile_2D.shape[1]-1] == 0, "Corner point should be zero"

    # edges should be equal
    assert np.array_equal(profile_2D[0, :], profile_2D[profile_2D.shape[0]-1, :]), "Edges should be equal (symmetric)"
    assert np.array_equal(profile_2D[:, 0], profile_2D[:, profile_2D.shape[1]-1]), "Edges should be equal (symmetric)"


    # edges should be zero
    assert np.array_equal(profile_2D[0, :], 0), "Edges should be zero (symmetric)"
    assert np.array_equal(profile_2D[:, 0], 0), "Edges should be zero (symmetric)"
    assert np.array_equal(profile_2D[profile_2D.shape[0]-1, :], 0), "Edges should be zero (symmetric)"
    assert np.array_equal(profile_2D[:, profile_2D.shape[1]-1], 0), "Edges should be zero (symmetric)"



    # maximum at midpoint
    midx, midy = profile_2D.shape[0] // 2, profile_2D.shape[1] // 2
    assert profile_2D[midx, midy] == np.max(profile_2D), "Maximum should be at the midpoint"