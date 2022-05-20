from dataclasses import dataclass
from turtle import distance
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from enum import Enum, auto

from lens_simulation import Lens, Simulation, utils, Medium

class BeamSpread(Enum):
    Plane = auto()
    Diverging = auto()
    Converging = auto()

class DistanceMode(Enum):
    Direct = auto()
    Width = auto()
    Focal = auto()

class BeamShape(Enum):
    Circular = auto()
    Square = auto()
    Rectangular = auto()

@dataclass
class BeamSettings:
    distance_mode: DistanceMode
    beam_spread: BeamSpread
    beam_shape: BeamShape
    width: float
    height: float
    position: list

def theta_from_NA(numerical_aperture: float, output_medium: Lens.Medium):

    return np.arcsin(numerical_aperture/output_medium.refractive_index)

def focal_distance_from_theta(beam: Lens, theta: float):
    return beam.diameter/2 / np.tan(theta)

def height_from_focal_distance(beam: Lens, output_medium: Lens.Medium, focal_distance: float):
    a = 1
    b = -2*focal_distance*(beam.medium.refractive_index-output_medium.refractive_index)/output_medium.refractive_index
    c = (beam.diameter/2)**2

    if (b**2 - 4*a*c < 0):
        raise ValueError("Negative value encountered in sqrt.  Can't find a lens height to give this focal distance")
    else: return (-b - np.sqrt(b**2 - 4*a*c))/(2*a)


class Beam:
    def __init__(self, settings: BeamSettings) -> None:

        self.settings: BeamSettings = settings

        self.distance_mode: DistanceMode
        self.spread: BeamSpread
        self.shape: BeamShape

        self.width: float
        self.height: float
        self.position: list[float]

        self.theta: float  # degrees
        self.source_distance: float
        self.final_width: float

        # non-rectangular beams are symmetric
        if self.settings.beam_shape is not BeamShape.Rectangular:
            self.settings.height = self.settings.width

        # plane waves only enabled for direct mode
        if self.settings.beam_spread is BeamSpread.Plane:
            self.settings.distance_mode = DistanceMode.Direct

        # can't do converging/divering square beams
        if self.settings.beam_spread is not BeamSpread.Plane:
            self.settings.beam_shape = BeamShape.Circular

        if self.settings.beam_spread is BeamSpread.Plane:
            self.final_width = self.settings.width





    def generate_profile(self) -> np.ndarray:

        return NotImplemented

