import pytest

from lens_simulation.Lens import Lens, LensType, GratingSettings
from lens_simulation.Medium import Medium
import numpy as np




# TODO: lens fixture
def generate_default_lens():
    # create lens
    lens = Lens(diameter=100e-6, 
                height=20e-6, 
                exponent=2.0, 
                medium=Medium(2.348))

    lens.generate_profile(1e-6, lens_type=LensType.Spherical)

    return lens

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
    assert np.allclose(profile_2D[0, :], 0), "Edges should be zero (symmetric)"
    assert np.allclose(profile_2D[:, 0], 0), "Edges should be zero (symmetric)"
    assert np.allclose(profile_2D[-1, :], 0), "Edges should be zero (symmetric)"
    assert np.allclose(profile_2D[:, -1], 0), "Edges should be zero (symmetric)"

    # maximum at midpoint
    midx, midy = profile_2D.shape[0] // 2, profile_2D.shape[1] // 2
    assert profile_2D[midx, midy] == np.max(profile_2D), "Maximum should be at the midpoint"



def test_lens_inverted():

    lens = generate_default_lens()

    lens.invert_profile()

    midx, midy = lens.profile.shape[0] // 2, lens.profile.shape[1] // 2

    assert np.isclose(np.min(lens.profile), lens.profile[midx, midy], atol=0.25e-6)
    assert np.isclose(np.max(lens.profile), lens.profile[0, 0], atol=0.25e-6)
    assert np.isclose(np.max(lens.profile), lens.profile[-1, -1], atol=0.25e-6)



def test_grating_mask():


    lens = generate_default_lens()

    grating_settings = GratingSettings(
        width = 20e-6,
        distance = 50e-6,
        depth = 1e-6,
        centred = True
    )
    lens.calculate_grating_mask(grating_settings, x_axis=True, y_axis=True)
    lens.apply_masks(grating=True)

    centre_x, centre_y = lens.profile.shape[0] // 2 , lens.profile.shape[1] // 2

    assert np.isclose(lens.profile[centre_x, centre_y], lens.height - grating_settings.depth, atol=0.25e-6), "Centre of lens should have grating"

def test_grating_mask_is_not_centred():

    lens = generate_default_lens()

    grating_settings = GratingSettings(
        width = 20e-6,
        distance = 50e-6,
        depth = 1e-6,
        centred = False
    )
    lens.calculate_grating_mask(grating_settings, x_axis=True, y_axis=True)
    lens.apply_masks(grating=True)

    centre_x, centre_y = lens.profile.shape[0] // 2 , lens.profile.shape[1] // 2

    assert np.isclose(lens.profile[centre_x, centre_y], lens.height, atol=0.25e-6), "Centre of lens should not have grating"


def test_grating_mask_raises_error():
    
    lens = generate_default_lens()

    grating_settings = GratingSettings(
        width = 10e-6,
        distance = 10e-6,
        depth = 1e-6,
        centred = True
    )

    with pytest.raises(ValueError):
        # distance between grating must be greater than grating width
        lens.calculate_grating_mask(grating_settings, x_axis=True, y_axis=True)
    
def test_truncation_by_value():

    truncation_value = 15e-6

    lens = generate_default_lens()

    lens.calculate_truncation_mask(truncation=truncation_value, type="value")
    lens.apply_masks(truncation=True)

    assert np.max(lens.profile) == truncation_value, "Maximum value should be truncation value"

def test_truncation_by_radius():

    truncation_radius = 25.0e-6

    lens = generate_default_lens()

    lens.calculate_truncation_mask(radius=truncation_radius, type="radial")
    lens.apply_masks(truncation=True)

    assert np.isclose(np.max(lens.profile), 15.e-6, atol=0.25e-6), "Maximum value should be 15e-6"


def test_apeture():

    inner_m = 0e-6
    outer_m = 25e-6

    lens = generate_default_lens()

    lens.calculate_apeture(inner_m = inner_m, outer_m=outer_m, type="radial", inverted=False) 
    lens.apply_masks(apeture=True)

    centre_x, centre_y = lens.profile.shape[0] // 2 , lens.profile.shape[1] // 2
    outer_px = int(outer_m / lens.pixel_size) - 1

    assert lens.profile[centre_x, centre_y] == 0, "Centre should be apetured"
    assert lens.profile[centre_x - outer_px, centre_y] == 0, "Outer radius should be apetured"
    assert lens.profile[centre_x + outer_px, centre_y] == 0, "Outer radius should be apetured"
    assert lens.profile[centre_x, centre_y - outer_px] == 0, "Outer radius should be apetured"
    assert lens.profile[centre_x, centre_y + outer_px] == 0, "Outer radius should be apetured"

def test_apeture_inverted():
    inner_m = 0e-6
    outer_m = 25e-6

    lens = generate_default_lens()

    lens.calculate_apeture(inner_m = inner_m, outer_m=outer_m, type="radial", inverted=True) 
    lens.apply_masks(apeture=True)

    centre_x, centre_y = lens.profile.shape[0] // 2 , lens.profile.shape[1] // 2
    outer_px = int(outer_m / lens.pixel_size) + 2

    assert np.isclose(lens.profile[centre_x, centre_y], lens.height, atol=0.25e-6), "Centre should be not apetured"
    assert lens.profile[centre_x - outer_px, centre_y] == 0, "Outer radius should be apetured"
    assert lens.profile[centre_x + outer_px, centre_y] == 0, "Outer radius should be apetured"
    assert lens.profile[centre_x, centre_y - outer_px] == 0, "Outer radius should be apetured"
    assert lens.profile[centre_x, centre_y + outer_px] == 0, "Outer radius should be apetured"

    assert lens.profile[0, 0] == 0, "Outer area should be apetured"
    assert lens.profile[0, -1] == 0, "Outer area should be apetured"
    assert lens.profile[-1, 0] == 0, "Outer area should be apetured"
    assert lens.profile[0, -1] == 0, "Outer area should be apetured"

# TODO: do the same tests for cylindrical....