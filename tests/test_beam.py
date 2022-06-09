import pytest
import numpy as np
from lens_simulation.beam import Beam,BeamSettings, DistanceMode, BeamSpread, BeamShape, focal_distance_from_theta, height_from_focal_distance
from lens_simulation.Lens import Lens, LensType
from lens_simulation.Medium import Medium
from lens_simulation.structures import SimulationParameters


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
        sim_height = 1500e-6,
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
    beam_settings.theta = 10

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

    return NotImplemented

def test_beam_generate_profile_plane_rectangular(beam_settings, sim_parameters):

    # plane - rectangular 
    beam_settings.beam_shape = BeamShape.Rectangular
    beam_settings.beam_spread = BeamSpread.Plane
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)
   

    return NotImplemented

def test_beam_generate_profile_converging(beam_settings, sim_parameters):

    # converging beam 
    beam_settings.beam_shape = BeamShape.Circular
    beam_settings.beam_spread = BeamSpread.Converging
    beam_settings.theta = 10
    beam = Beam(beam_settings)

    return NotImplemented


def test_beam_generate_profile_diverging(beam_settings, sim_parameters):

    # diverging beam
    beam_settings.beam_shape = BeamShape.Circular
    beam_settings.beam_spread = BeamSpread.Diverging
    beam_settings.theta = 10
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    # profile is inverted, max should be zero
    # assert np.isclose(np.max(beam.lens.profile), 0)


def test_beam_propagation_distance_direct(beam_settings, sim_parameters):

    test_distance = 10.e-3
    beam_settings.distance_mode = DistanceMode.Direct
    beam_settings.beam_spread = BeamSpread.Plane
    
    beam_settings.source_distance = test_distance 
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    sd, fd = beam.calculate_propagation_distance()

    assert sd == 0
    assert fd == test_distance, "finish distance for DistanceMode.Direct should be the same as source_distance"

def test_beam_propagation_distance_plane(beam_settings, sim_parameters):

    test_distance = 10.e-3
    beam_settings.distance_mode = DistanceMode.Focal
    beam_settings.beam_spread = BeamSpread.Plane
    beam_settings.source_distance = test_distance 
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    sd, fd = beam.calculate_propagation_distance()

    assert sd == 0
    assert fd == test_distance, "finish distance for BeamSpread.Plane should be the same as source_distance"


def test_beam_propagation_distance_focal(beam_settings, sim_parameters):

    test_distance = 10.e-3
    beam_settings.distance_mode = DistanceMode.Focal
    beam_settings.beam_spread = BeamSpread.Converging
    beam_settings.theta = 10
    beam_settings.source_distance = test_distance 
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    sd, fd = beam.calculate_propagation_distance()

    # focal_distance = focal_distance_from_theta(beam=beam.lens, theta=beam_settings.theta)

    assert beam.distance_mode == DistanceMode.Focal
    assert sd == 0
    assert fd == beam.focal_distance, "finish distance for DistanceMode.Focal should be the same as focal_distance"

def test_beam_propagation_distance_width(beam_settings, sim_parameters):

    test_distance = 10.e-3
    beam_settings.distance_mode = DistanceMode.Width
    beam_settings.beam_spread = BeamSpread.Converging
    beam_settings.theta = 10
    beam_settings.final_width = 5e-6
    beam_settings.source_distance = test_distance 
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    sd, fd = beam.calculate_propagation_distance()

    # assert beam.distance_mode == DistanceMode.Width
    # assert sd == 0
    # assert fd == beam.focal_distance, "finish distance for DistanceMode.Focal should be the same as focal_distance"