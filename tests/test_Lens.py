import pytest

from lens_simulation.Lens import Lens, LensType, GratingSettings
from lens_simulation.Medium import Medium
import numpy as np


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


def test_create_lens(cylindrical_lens):

    lens = cylindrical_lens

    assert lens.diameter == LENS_DIAMETER
    assert lens.height == LENS_HEIGHT
    assert lens.exponent == LENS_FOCUS_EXPONENT


def test_axicon_lens(cylindrical_lens):

    lens = Lens(diameter=LENS_DIAMETER,
                height=LENS_HEIGHT,
                exponent=LENS_AXICON_EXPONENT,
                medium=Medium(2.348))

    lens.generate_profile(LENS_PIXEL_SIZE, lens_type=LensType.Cylindrical)


    profile = lens.generate_profile(pixel_size=100e-9, lens_type=LensType.Cylindrical)

    assert np.isclose(np.min(profile), 0, atol=5e-7)  # TODO: broken
    assert np.isclose(np.max(profile), lens.height,  atol=5e-7)
    assert np.isclose(profile[:, int(profile.shape[-1] * 0.75)],  lens.height / 2, atol=5e-7)


def test_focusing_lens(cylindrical_lens):

    lens = cylindrical_lens
    profile = lens.generate_profile(pixel_size=100e-9, lens_type=LensType.Cylindrical)

    assert np.isclose(0, np.min(profile), atol=5e-7)
    assert np.isclose(np.max(profile), lens.height, atol=5e-7)
    assert not profile[:, int(profile.shape[-1] * 0.75)] == -lens.height / 2


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


def test_lens_inverted(spherical_lens):

    lens = spherical_lens

    lens.invert_profile()

    midx, midy = lens.profile.shape[0] // 2, lens.profile.shape[1] // 2

    assert np.isclose(np.min(lens.profile), lens.profile[midx, midy], atol=0.25e-6)
    assert np.isclose(np.max(lens.profile), lens.profile[0, 0], atol=0.25e-6)
    assert np.isclose(np.max(lens.profile), lens.profile[-1, -1], atol=0.25e-6)



def test_grating_mask(spherical_lens):


    lens = spherical_lens

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

def test_grating_mask_is_not_centred(spherical_lens):

    lens = spherical_lens

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


def test_grating_mask_raises_error(spherical_lens):

    lens = spherical_lens

    grating_settings = GratingSettings(
        width = 10e-6,
        distance = 10e-6,
        depth = 1e-6,
        centred = True
    )

    with pytest.raises(ValueError):
        # distance between grating must be greater than grating width
        lens.calculate_grating_mask(grating_settings, x_axis=True, y_axis=True)

def test_truncation_by_value(spherical_lens):

    truncation_value = 15e-6

    lens = spherical_lens

    lens.create_truncation_mask(truncation_height=truncation_value, type="value")
    lens.apply_masks(truncation=True)

    assert np.max(lens.profile) == truncation_value, "Maximum value should be truncation value"

def test_truncation_by_radius(spherical_lens):

    truncation_radius = 25.0e-6

    lens = spherical_lens

    lens.create_truncation_mask(radius=truncation_radius, type="radial")
    lens.apply_masks(truncation=True)

    assert np.isclose(np.max(lens.profile), 15.e-6, atol=0.25e-6), "Maximum value should be 15e-6"


def test_aperture(spherical_lens):

    inner_m = 0e-6
    outer_m = 25e-6

    lens = spherical_lens

    lens.calculate_aperture(inner_m = inner_m, outer_m=outer_m, type="radial", inverted=False)
    lens.apply_masks(aperture=True)

    centre_x, centre_y = lens.profile.shape[0] // 2 , lens.profile.shape[1] // 2
    outer_px = int(outer_m / lens.pixel_size) - 1

    assert lens.profile[centre_x, centre_y] == 0, "Centre should be apertured"
    assert lens.profile[centre_x - outer_px, centre_y] == 0, "Outer radius should be apertured"
    assert lens.profile[centre_x + outer_px, centre_y] == 0, "Outer radius should be apertured"
    assert lens.profile[centre_x, centre_y - outer_px] == 0, "Outer radius should be apertured"
    assert lens.profile[centre_x, centre_y + outer_px] == 0, "Outer radius should be apertured"

def test_aperture_inverted(spherical_lens):
    inner_m = 0e-6
    outer_m = 25e-6

    lens = spherical_lens

    lens.calculate_aperture(inner_m = inner_m, outer_m=outer_m, type="radial", inverted=True)
    lens.apply_masks(aperture=True)

    centre_x, centre_y = lens.profile.shape[0] // 2 , lens.profile.shape[1] // 2
    outer_px = int(outer_m / lens.pixel_size) + 2

    assert np.isclose(lens.profile[centre_x, centre_y], lens.height, atol=0.25e-6), "Centre should be not apertured"
    assert lens.profile[centre_x - outer_px, centre_y] == 0, "Outer radius should be apertured"
    assert lens.profile[centre_x + outer_px, centre_y] == 0, "Outer radius should be apertured"
    assert lens.profile[centre_x, centre_y - outer_px] == 0, "Outer radius should be apertured"
    assert lens.profile[centre_x, centre_y + outer_px] == 0, "Outer radius should be apertured"

    assert lens.profile[0, 0] == 0, "Outer area should be apertured"
    assert lens.profile[0, -1] == 0, "Outer area should be apertured"
    assert lens.profile[-1, 0] == 0, "Outer area should be apertured"
    assert lens.profile[0, -1] == 0, "Outer area should be apertured"
