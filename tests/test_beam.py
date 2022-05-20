import pytest
import numpy as np
from lens_simulation.beam import Beam,BeamSettings, DistanceMode, BeamSpread, BeamShape, focal_distance_from_theta, height_from_focal_distance
from lens_simulation.Lens import Lens, LensType
from lens_simulation.Medium import Medium
from lens_simulation.SimulationUtils import SimulationParameters


@pytest.fixture
def beam_settings():

    return BeamSettings(
        distance_mode=DistanceMode.Direct,
        beam_spread=BeamSpread.Plane, 
        beam_shape=BeamShape.Square,
        width= 10e-6,
        height= 5e-6,
        position=[-0e-6, 0e-6],
        source_distance=10e-6
    )

@pytest.fixture
def sim_parameters():

    return SimulationParameters(
        A = 10000, 
        pixel_size=1e-6, 
        sim_wavelength = 488e-9,
        sim_width = 1500e-6,
        lens_type=LensType.Spherical
    )


def test_beam_is_symmetric(beam_settings):
    """Non-rectangular beams should be symmetric"""

    # square
    beam = Beam(beam_settings)
    assert beam.settings.width == beam.settings.height

    # circular
    beam_settings.beam_shape = BeamShape.Circular
    beam = Beam(beam_settings)
    assert beam.settings.width == beam.settings.height


def test_beam_plane_wave_distance_mode(beam_settings):

    beam_settings.beam_spread = BeamSpread.Plane
    beam_settings.distance_mode = DistanceMode.Focal
    beam = Beam(beam_settings)

    assert beam.settings.beam_spread == BeamSpread.Plane
    assert beam.settings.distance_mode == DistanceMode.Direct

def test_beam_converging_is_circular(beam_settings):

    beam_settings.beam_shape = BeamShape.Square
    beam_settings.beam_spread = BeamSpread.Converging

    beam = Beam(beam_settings)

    assert beam.settings.beam_spread == BeamSpread.Converging
    assert beam.settings.beam_shape == BeamShape.Circular


def test_beam_plane_wave_has_constant_width(beam_settings):

    beam_settings.beam_spread = BeamSpread.Plane
    beam = Beam(beam_settings)

    assert beam.final_width == beam.settings.width


def test_beam_generate_profile_plane_square(beam_settings, sim_parameters):

    # plane - square
    beam_settings.beam_shape = BeamShape.Square
    beam_settings.beam_spread = BeamSpread.Plane
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    centre_x, centre_y = beam.lens.profile.shape[1] // 2, beam.lens.profile.shape[0] // 2, 
    assert np.isclose(beam.lens.profile[centre_y, centre_x], 1.0)


def test_beam_generate_profile_plane_circular(beam_settings, sim_parameters):

    # plane - circular, 
    beam_settings.beam_shape = BeamShape.Circular
    beam_settings.beam_spread = BeamSpread.Plane
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    # TODO
    return NotImplemented

def test_beam_generate_profile_plane_rectangular(beam_settings, sim_parameters):

    # plane - rectangular 
    beam_settings.beam_shape = BeamShape.Rectangular
    beam_settings.beam_spread = BeamSpread.Plane
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)
   
    # TODO
    return NotImplemented

def test_beam_generate_profile_converging(beam_settings, sim_parameters):

    # converging beam 
    beam_settings.beam_shape = BeamShape.Circular
    beam_settings.beam_spread = BeamSpread.Converging
    beam = Beam(beam_settings)

    return NotImplemented
    # TODO: 
    # lens = Lens(
    #     diameter=max(beam.settings.width, beam.settings.height),
    #     height=100,
    #     exponent=2,
    #     medium = Medium(100)
    # )

    # # calculate height from focal distance
    # focal_distance = focal_distance_from_theta(beam=lens, theta=beam.theta)
    # equivalent_height = height_from_focal_distance(lens, output_medium=output_medium, focal_distance=focal_distance)
    
    # beam.generate_profile()

    # assert np.isclose(beam.lens.height, equivalent_height)
    # assert np.isclose(np.max(beam.lens.profile), equivalent_height)


def test_beam_generate_profile_diverging(beam_settings, sim_parameters):

    # diverging beam
    beam_settings.beam_shape = BeamShape.Circular
    beam_settings.beam_spread = BeamSpread.Diverging
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    # profile is inverted, max should be zero
    # assert np.isclose(np.max(beam.lens.profile), 0)  # TODO


def test_validate_beam_configuration(beam_settings):

    # TODO:
    return NotImplemented