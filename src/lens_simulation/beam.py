from dataclasses import dataclass, field
from xml.dom import ValidationErr
from jsonschema import ValidationError
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from enum import Enum, auto

from lens_simulation import utils
from lens_simulation.Medium import Medium 
from lens_simulation.Lens import Lens, LensType
from lens_simulation.SimulationUtils import SimulationParameters

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

############
# Bsp/ DM:      Direct      Width       Focal   
# Plane:          Y                       N
# Converging:     Y                       Y
# Diverging:      Y                       Y



@dataclass
class BeamSettings:
    distance_mode: DistanceMode
    beam_spread: BeamSpread
    beam_shape: BeamShape
    width: float
    height: float
    position: list = field(default_factory=[0, 0])
    theta: float = 0.0                      # degrees
    numerical_aperture: float = None        # ?
    tilt: float = None                      # degrees
    source_distance: float = None
    final_width: float = None
    focal_multiple: float = None



class Beam:
    def __init__(self, settings: BeamSettings) -> None:

        # validate the beam configuration
        settings = validate_beam_configuration(settings)

        self.distance_mode: DistanceMode = settings.distance_mode
        self.spread: BeamSpread = settings.beam_spread
        self.shape: BeamShape = settings.beam_shape

        self.width: float = settings.width
        self.height: float  = settings.height
        self.position: list[float] = settings.position

        self.theta: float  = np.deg2rad(settings.theta) # degrees -> rad
        self.source_distance: float = settings.source_distance
        self.final_width: float = settings.final_width

        self.output_medium = Medium(1.33)

        self.settings: BeamSettings = settings # TODO: remove?


    def generate_profile(self, sim_parameters: SimulationParameters) -> np.ndarray:

        pixel_size = sim_parameters.pixel_size
        sim_width = sim_parameters.sim_width
        lens_type = sim_parameters.lens_type

        # Default beam specifications
        lens = Lens(
            diameter=max(self.width, self.height),
            height=100,                     # arbitrary non-zero
            exponent=2,                     # must be 2 for focusing
            medium=Medium(100)              # arbitrary non-zero
        )

        lens.generate_profile(pixel_size=pixel_size, lens_type=lens_type)

        # generate the lens profile
        if self.spread is BeamSpread.Plane:

            if self.shape is BeamShape.Circular:
                lens.profile = (lens.profile != 0) * 1 # removes corner areas from lens
            
            if self.shape is BeamShape.Square:
                lens.profile = np.ones(shape=lens.profile.shape)
            
            if self.shape is BeamShape.Rectangular:
                lens.profile = np.zeros(shape=lens.profile.shape)

                # make sure to fill out in the correct order, otherwise this creates a square        
                if self.height > self.width:
                    profile_width = int(self.width/pixel_size/2)   # half width
                    lens.profile[lens.profile.shape[0]//2-profile_width:lens.profile.shape[0]//2+profile_width, :] = 1
            
                elif self.width > self.height:
                    profile_height = int(self.height/pixel_size/2) # half height
                    lens.profile[:, lens.profile.shape[0]//2-profile_height:lens.profile.shape[0]//2+profile_height] = 1

        # diverging/converging cases
        elif self.spread in [BeamSpread.Converging, BeamSpread.Diverging]:
            # calculate the equivalent focal distance of the required convergence angle
            self.focal_distance = focal_distance_from_theta(beam=lens, theta=self.theta)
            
            # calculate and set the height of the apertures 'virtual' lens, re-generate the profile with new height
            lens.height = height_from_focal_distance(lens, output_medium=self.output_medium, focal_distance=self.focal_distance)

            # regenerate lens profile
            lens.generate_profile(pixel_size=pixel_size, lens_type=lens_type)

            if self.spread is BeamSpread.Diverging:
                lens.invert_profile()
                lens.profile = (lens.profile < lens.height) * lens.profile

        else:
            raise TypeError(f"Unsupport Beam Spread: {self.spread}")



        # set up the part of the lens square that isn't the lens for aperturing
        non_lens_profile = lens.profile == 0 
        aperturing_value = -1e-9
        lens.profile[non_lens_profile] = aperturing_value

        # apeturing profile
        self.non_lens_profile = non_lens_profile
        

        # calculate padding parameters
        beam_position = self.position
        pad_width = (int(sim_width/pixel_size)-lens.profile.shape[0])//2 + 1 
        relative_position_x = int(beam_position[1]/pixel_size)
        relative_position_y = int(beam_position[0]/pixel_size)

        # pad the profile to the sim width (Top - Bottom - Left - Right)
        lens.profile = np.pad(lens.profile, ((pad_width + relative_position_y, pad_width - relative_position_y),
                                                    (pad_width + relative_position_x, pad_width - relative_position_x)), 
                                                    mode="constant", constant_values=aperturing_value)

        # assign lens
        self.lens = lens


    def calculate_propagation_distance(self):
        """Calculate the total distance to propagate the beam depending on configuration"""

        start_distance = 0

        # If you just want the source to be a certain distance away:
        if self.distance_mode is DistanceMode.Direct or self.spread is BeamSpread.Plane:
            finish_distance = self.source_distance

        elif self.distance_mode is DistanceMode.Focal:
            finish_distance = self.focal_distance

        # if you want the beam to converge/diverge to a specific width
        elif self.distance_mode is DistanceMode.Width:
            final_beam_radius = self.final_width/2
            if self.spread is BeamSpread.Converging:
                finish_distance = self.focal_distance - (final_beam_radius/np.tan(self.theta))
            else:
                finish_distance = ((final_beam_radius-(self.lens.diameter/2))/np.tan(self.theta))
        else:
            raise TypeError(f"Unsupported DistanceMode for calculated propagation distance: {self.distance_mode}")

        return start_distance, finish_distance

    
def calculate_tilted_delta_profile(lens: Lens, output_medium: Medium, tilt_enabled: bool = False, xtilt: float = 0, ytilt: float = 0) -> np.ndarray:
    """_summary_

    Args:
        lens (Lens): lens
        output_medium (Medium): output medium
        tilt_enabled (bool, optional): delta profile is tilted. Defaults to False.
        xtilt (float, optional): tilt in x-axis (degrees). Defaults to 0.
        ytilt (float, optional): tilt in y-axis (degrees). Defaults to 0.

    Returns:
        np.ndarray: delta profile
    """

    # regular delta calculation
    delta = (lens.medium.refractive_index-output_medium.refractive_index) * lens.profile

    # tilt the beam
    if tilt_enabled:
        x = np.arange(len(lens.profile))*lens.pixel_size
        y = np.arange(len(lens.profile))*lens.pixel_size

        # modify the optical path of the light based on tilt
        delta = delta + np.add.outer(y * np.tan(np.deg2rad(ytilt)), -x * np.tan(np.deg2rad(xtilt)))

    return delta



def validate_beam_configuration(settings: BeamSettings):
    """Validate the user has passed the correct parameters for the given configuration"""
    # beam spread
    if settings.beam_spread is BeamSpread.Plane:

        if settings.source_distance is None:
            raise ValidationError("A source_distance must be provided for BeamSpread.Plane")

        # plane wave is constant width along optical axis
        settings.final_width = settings.width
        print(f"The plane wave if constant along the optical axis. The beam final_width has been set to the initial width: {settings.width:.2e}m")

        # plane waves only enabled for direct mode
        settings.distance_mode = DistanceMode.Direct
        print(f"Only DistanceMode.Direct is supported for BeamSpread.Plane. The distance_mode has been set to {settings.distance_mode}.")

    # can't do converging/divering square beams
    if settings.beam_spread in [BeamSpread.Converging, BeamSpread.Diverging]:
        settings.beam_shape = BeamShape.Circular
        print(f"Only BeamShape.Circular is supported for {settings.beam_spread}. The beam_shape has been set to {settings.beam_shape}.")

    # beam shape
    # non-rectangular beams are symmetric
    if settings.beam_shape in [BeamShape.Circular, BeamShape.Square]:
        settings.height = settings.width
        print(f"The beam_shape ({settings.beam_shape}) requires a symmetric beam. The beam height has been set to the beam width: {settings.width:.2e}m ")

    # distance mode
    if settings.distance_mode == DistanceMode.Direct:
        if settings.source_distance is None:
            raise ValidationError("A source_distance must be provided for DistanceMode.Direct")

    if settings.distance_mode == DistanceMode.Focal:
        if settings.beam_spread not in [BeamSpread.Converging, BeamSpread.Diverging]:
            raise ValidationError(f"BeamSpread must be Converging, or Diverging for DistanceMode.Focal (currently {settings.beam_spread})")

    if settings.distance_mode == DistanceMode.Width:
        if settings.final_width is None:
            raise ValidationError(f"A final_width must be provided for DistanceMode.Width")

    return settings

def theta_from_NA(numerical_aperture: float, output_medium: Medium):

    return np.arcsin(numerical_aperture/output_medium.refractive_index)

def focal_distance_from_theta(beam: Lens, theta: float):
    return beam.diameter/2 / np.tan(theta + 1e-12)

def height_from_focal_distance(beam: Lens, output_medium: Medium, focal_distance: float):
    a = 1
    b = -2*focal_distance*(beam.medium.refractive_index-output_medium.refractive_index)/output_medium.refractive_index
    c = (beam.diameter/2)**2

    if (b**2 - 4*a*c < 0):
        raise ValueError("Negative value encountered in sqrt.  Can't find a lens height to give this focal distance")
    else: return (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
