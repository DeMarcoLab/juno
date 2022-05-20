from turtle import width
import pytest
from lens_simulation.beam import Beam,BeamSettings, DistanceMode, BeamSpread, BeamShape

@pytest.fixture
def beam_settings():

    return BeamSettings(
        distance_mode=DistanceMode.Direct,
        beam_spread=BeamSpread.Plane, 
        beam_shape=BeamShape.Square,
        width= 10e-6,
        height= 5e-6,
        position=[-0e-6, 0e-6]
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