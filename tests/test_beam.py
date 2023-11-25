import pytest
import numpy as np
from juno.beam import Beam,BeamSettings, DistanceMode, BeamSpread, BeamShape, focal_distance_from_theta, height_from_focal_distance, validate_beam_configuration
from juno.Lens import Lens, LensType
from juno.Medium import Medium
from juno.structures import SimulationParameters


@pytest.fixture
def beam_settings():

    return BeamSettings(
        distance_mode=DistanceMode.Direct,
        beam_spread=BeamSpread.Plane, 
        beam_shape=BeamShape.Rectangular,
        width= 10e-6,
        height= 5e-6,
        position_x=0,
        position_y=0,
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

    beam_settings.beam_shape = BeamShape.Rectangular
    beam_settings.beam_spread = BeamSpread.Converging
    beam_settings.theta = 10

    beam = Beam(beam_settings)

    assert beam.settings.beam_spread == BeamSpread.Converging
    assert beam.settings.beam_shape == BeamShape.Circular


def test_beam_plane_wave_has_constant_width(beam_settings):

    beam_settings.beam_spread = BeamSpread.Plane
    beam = Beam(beam_settings)

    assert beam.final_diameter == beam.settings.width


def test_beam_generate_profile_plane_square(beam_settings, sim_parameters):

    # plane - square
    beam_settings.beam_shape = BeamShape.Rectangular
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
    beam_settings.distance_mode = DistanceMode.Diameter
    beam_settings.beam_spread = BeamSpread.Converging
    beam_settings.theta = 10
    beam_settings.final_diameter = 5e-6
    beam_settings.source_distance = test_distance 
    beam = Beam(beam_settings)
    beam.generate_profile(sim_parameters)

    sd, fd = beam.calculate_propagation_distance()

    # assert beam.distance_mode == DistanceMode.Diameter
    # assert sd == 0
    # assert fd == beam.focal_distance, "finish distance for DistanceMode.Focal should be the same as focal_distance"



def test_generate_profile_raises_error(beam_settings, sim_parameters):


    parameters = sim_parameters
    beam = Beam(beam_settings)

    # beam width greater than sim
    parameters.sim_width = beam.width / 2.0
    with pytest.raises(ValueError):
        beam.generate_profile(parameters)

    # beam height greater than sim
    parameters = sim_parameters
    parameters.sim_height = beam.height / 2.0
    with pytest.raises(ValueError):
        beam.generate_profile(parameters)

    # beam position outside sim
    parameters = sim_parameters
    beam.position = (parameters.sim_width * 2, parameters.sim_height * 2)
    with pytest.raises(ValueError):
        beam.generate_profile(parameters)

def test_validate_beam_configuration(beam_settings):

    # plane beam with no source distance
    settings = beam_settings
    settings.beam_spread = BeamSpread.Plane
    settings.source_distance = None

    with pytest.raises(ValueError):
        validate_beam_configuration(settings)

    # converging beam with no theta or numerical aperture
    settings.beam_spread = BeamSpread.Converging
    settings.theta = 0.0
    settings.numerical_aperture = None
    with pytest.raises(ValueError):
        validate_beam_configuration(settings)

    # distance mode direct with no source distance
    settings.distance_mode = DistanceMode.Direct
    settings.source_distance = None
    with pytest.raises(ValueError):
        validate_beam_configuration(settings)

    # distance mode focal with no convergence / divergence
    settings.distance_mode = DistanceMode.Focal
    settings.beam_spread = BeamSpread.Plane
    with pytest.raises(ValueError):
        validate_beam_configuration(settings)

    # distance mode diameter with no diameter
    settings.distance_mode = DistanceMode.Diameter
    settings.final_diameter = None
    with pytest.raises(ValueError):
        validate_beam_configuration(settings)

    # distance mode direct with step size greater than source distance
    settings.distance_mode = DistanceMode.Direct
    settings.source_distance = 1e-3
    settings.step_size = 2e-3
    with pytest.raises(ValueError):
        validate_beam_configuration(settings)




