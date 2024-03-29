import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from juno import utils, validation
from juno.Lens import Lens, LensType
from juno.Medium import Medium
from juno.structures import SimulationParameters


class BeamSpread(Enum):
    Plane = auto()
    Diverging = auto()
    Converging = auto()

class DistanceMode(Enum):
    Direct = auto()
    Diameter = auto()
    Focal = auto()

class BeamShape(Enum):
    Circular = auto()
    Rectangular = auto()

class BeamOperator(Enum):
    Plane = auto()
    Gaussian = auto()
    Custom = auto()

############
# Bsp/ DM:      Direct      Diameter       Focal
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
    position_x: 0.0
    position_y: 0.0
    theta: float = 0.0                      # degrees
    numerical_aperture: float = None
    tilt_x: float = 0.0
    tilt_y: float = 0.0                     # degrees
    source_distance: float = None
    final_diameter: float = None
    focal_multiple: float = None
    n_steps: int = 10
    step_size: float = None
    output_medium: float = 1.0
    operator: BeamOperator = BeamOperator.Plane
    gaussian_wx: float = None
    gaussian_wy: float = None
    gaussian_z0: float = None
    gaussian_z: float = None
    data: str = None # path to custom array

    @staticmethod
    def __from__dict__(config: dict) -> 'BeamSettings':
        """Load the beam settings from dictionary

        Args:
            config (dict): beam configuration as dictionary

        Returns:
            BeamSettings: beam configuration as BeamSettings
        """

        config = validation._validate_default_beam_config(config)

        beam_settings = BeamSettings(
            distance_mode=DistanceMode[config["distance_mode"]],
            beam_spread=BeamSpread[config["spread"]],
            beam_shape=BeamShape[config["shape"]],
            width=config["width"],
            height= config["height"],
            position_x=config["position_x"],
            position_y=config["position_y"],
            theta=config["theta"],
            numerical_aperture=config["numerical_aperture"],
            tilt_x=config["tilt_x"],
            tilt_y=config["tilt_y"],
            source_distance=config["source_distance"],
            final_diameter=config["final_diameter"],
            focal_multiple=config["focal_multiple"],
            n_steps=config["n_steps"],
            step_size=config["step_size"],
            output_medium=config["output_medium"],
            operator=BeamOperator[config["operator"]],
            gaussian_wx=config["gaussian_wx"],
            gaussian_wy=config["gaussian_wy"],
            gaussian_z0=config["gaussian_z0"],
            gaussian_z=config["gaussian_z"],
            data=config["data"]
        )

        return beam_settings


class Beam:
    def __init__(self, settings: BeamSettings) -> None:

        # validate the beam configuration
        settings = validate_beam_configuration(settings)

        self.distance_mode: DistanceMode = settings.distance_mode
        self.spread: BeamSpread = settings.beam_spread
        self.shape: BeamShape = settings.beam_shape

        self.width: float = settings.width
        self.height: float  = settings.height
        self.position: list[float] = [settings.position_x, settings.position_y]

        self.theta: float  = np.deg2rad(settings.theta) # degrees -> rad
        self.source_distance: float = settings.source_distance
        self.final_diameter: float = settings.final_diameter

        self.tilt: list[float] = [settings.tilt_x, settings.tilt_y]

        self.settings: BeamSettings = settings

    def __repr__(self) -> str:

        return f"""Beam: {self.settings}"""


    def generate_profile(self, sim_parameters: SimulationParameters) -> np.ndarray:

        pixel_size = sim_parameters.pixel_size
        sim_width = sim_parameters.sim_width
        sim_height = sim_parameters.sim_height
        self.output_medium = Medium(self.settings.output_medium, sim_parameters.sim_wavelength)

        # validation
        if self.width > sim_width:
            raise ValueError(f"Beam width is larger than simulation width: beam={self.width:.2e}m, sim={sim_width:.2e}m")
        if self.height > sim_height:
            raise ValueError(f"Beam height is larger than simulation height: beam={self.height:.2e}m, sim={sim_height:.2e}m")

        if self.position[0] > sim_width or self.position[1] > sim_width:
            raise ValueError(f"Beam position is outside simulation: position: x:{self.position[0]:.2e}m, y:{self.position[1]:.2e}m, sim_size: {sim_width:.2e}m")

        # Default beam specifications
        if self.shape is BeamShape.Rectangular:
            lens_type = LensType.Cylindrical
        if self.shape is BeamShape.Circular:
            lens_type = LensType.Spherical

        lens = Lens(
            diameter=max(self.width, self.height),
            height=100,                     # arbitrary non-zero
            exponent=2,                     # must be 2 for focusing
            medium=Medium(100),                  # arbitrary non-zero
            lens_type=lens_type
        )

        lens.generate_profile(pixel_size=pixel_size)

        # generate the lens profile
        if self.spread is BeamSpread.Plane:

            if self.shape is BeamShape.Circular:
                lens.profile = (lens.profile != 0) * 1 # removes corner areas from lens

            if self.shape is BeamShape.Rectangular:

                height_px = utils._calculate_num_of_pixels(self.height, pixel_size, odd=True)
                width_px = utils._calculate_num_of_pixels(self.width, pixel_size, odd=True)
                lens.profile = np.ones(shape=(height_px, width_px))

        # diverging/converging cases
        elif self.spread in [BeamSpread.Converging, BeamSpread.Diverging]:

            if self.theta == 0.0:
                self.theta = theta_from_NA(self.settings.numerical_aperture, self.output_medium)

            # calculate the equivalent focal distance of the required convergence angle
            self.focal_distance = focal_distance_from_theta(beam=lens, theta=self.theta)

            # calculate and set the height of the apertures 'virtual' lens, re-generate the profile with new height
            lens.height = height_from_focal_distance(lens, output_medium=self.output_medium, focal_distance=self.focal_distance)

            # regenerate lens profile
            lens.generate_profile(pixel_size=pixel_size)

            if self.spread is BeamSpread.Diverging:
                lens.invert_profile()
                lens.profile = (lens.profile < lens.height) * lens.profile

        else:
            raise TypeError(f"Unsupport Beam Spread: {self.spread}")


        # position and pad beam
        self.position_and_pad_beam(lens, sim_parameters)

        # calculate propagation distance
        self.start_distance, self.finish_distance = self.calculate_propagation_distance()

    def position_and_pad_beam(self, lens: Lens, parameters: SimulationParameters):

        pixel_size = parameters.pixel_size
        sim_width = parameters.sim_width
        sim_height = parameters.sim_height
        sim_width_px = utils._calculate_num_of_pixels(sim_width, pixel_size)
        sim_height_px = utils._calculate_num_of_pixels(sim_height, pixel_size)

        # set up the part of the lens profile that isn't the lens for aperturing
        non_lens_profile = lens.profile == 0
        aperturing_value = -1 #-1e-9 # NOTE: needs to be a relatively large value for numerical comparison # QUERY why not zero, parts of lens at zero might not be apertures e.g. escape path...
        lens.profile[non_lens_profile] = aperturing_value # the non-lens-profile is the size of the lens before padding...

        # calculate padding parameters
        beam_position = self.position

        # when sim height 2px less than sim_width
        pad_height = (sim_height_px-lens.profile.shape[0])//2   # px
        pad_width = (sim_width_px-lens.profile.shape[1])//2    # px

        relative_position_x = int(beam_position[0]/pixel_size)                  # px
        relative_position_y = int(beam_position[1]/pixel_size)                  # px

        # pad the profile to the sim width (Top - Bottom - Left - Right)
        lens.profile = np.pad(lens.profile, ((pad_height + relative_position_y, pad_height - relative_position_y),
                                                    (pad_width + relative_position_x, pad_width - relative_position_x)),
                                                    mode="constant", constant_values=aperturing_value)
        # assign lens
        self.lens = lens
        self.lens.aperture = (lens.profile == aperturing_value).astype(bool)

        # reset apertures back to zero height
        self.lens.profile[self.lens.aperture] = 0
        # TODO: this should be applied with apply_apertures...


    def calculate_propagation_distance(self):
        """Calculate the total distance to propagate the beam depending on configuration"""

        # distance_mode
        # beam_spread
        # source_distance
        # focal_distance
        # final_diameter
        # theta
        # lens

        start_distance = 0

        # If you just want the source to be a certain distance away:
        if self.distance_mode is DistanceMode.Direct or self.spread is BeamSpread.Plane:
            finish_distance = self.source_distance

        elif self.distance_mode is DistanceMode.Focal:
            finish_distance = self.focal_distance * self.settings.focal_multiple

        # if you want the beam to converge/diverge to a specific width
        elif self.distance_mode is DistanceMode.Diameter:
            final_beam_radius = self.final_diameter/2
            if self.spread is BeamSpread.Converging:
                finish_distance = self.focal_distance - (final_beam_radius/np.tan(self.theta+1e-12))
            else:
                finish_distance = ((final_beam_radius-(self.lens.diameter/2))/np.tan(self.theta+1e-12))
        else:
            raise TypeError(f"Unsupported DistanceMode for calculated propagation distance: {self.distance_mode}")

        return start_distance, finish_distance

    def generate_wavefront(self, parameters: SimulationParameters) -> np.ndarray:
        """Generate the initial wavefront."""

        if self.settings.operator is BeamOperator.Gaussian:

            z0 = self.settings.gaussian_z0
            r0 = (0, 0)
            w0 = self.settings.gaussian_wx
            self.wavefront = create_gaussian(r0, w0, z0, parameters=parameters, theta=0, phi=0)
        
        if self.settings.operator is BeamOperator.Custom:

            fname = self.settings.data
            self.wavefront = utils.load_np_arr(fname) 

        if self.settings.operator is BeamOperator.Plane: 
            self.wavefront = np.ones_like(self.lens.profile)

        return self.wavefront


def create_gaussian(r0: tuple, w0: float, z0: float, parameters:SimulationParameters, theta:float = 0, phi: float = 0) -> np.ndarray:

    # TODO: @david reference?
    wavelength, A = parameters.sim_wavelength, parameters.A

    px_x = utils._calculate_num_of_pixels(parameters.sim_width,parameters.pixel_size, odd=True)
    px_y = utils._calculate_num_of_pixels(parameters.sim_height,parameters.pixel_size, odd=True)

    x = np.linspace(-parameters.sim_width / 2, parameters.sim_width / 2, px_x)
    y = np.linspace(-parameters.sim_height / 2, parameters.sim_height / 2, px_y)
    X, Y = np.meshgrid(x, y)

    if isinstance(w0, (float, int, complex)):
        w0 = (w0, w0)

    w0x, w0y = w0
    w0 = np.sqrt(w0x * w0y)
    x0, y0 = r0
    k = 2 * np.pi / wavelength

    # only for x axis.
    z_rayleigh = k * w0x**2 / 2

    phaseGouy = np.arctan2(z0, z_rayleigh)

    wx = w0x * np.sqrt(1 + (z0 / z_rayleigh)**2)
    wy = w0y * np.sqrt(1 + (z0 / z_rayleigh)**2)
    w = np.sqrt(wx * wy)

    if z0 == 0:
        R = 1e10
    else:
        R = z0 * (1 + (z_rayleigh / z0)**2)

    amplitude = A * w0 / w * np.exp(-(x0-X)**2 / (wx**2) -
                                    (y0-Y)**2 / (wy**2))

    phase1 = np.exp(1.j * k * (X * np.sin(theta) * np.cos(phi) +
                            Y * np.cos(theta) * np.sin(phi))) 

    # ??
    phase2 = np.exp(-1j * (k * z0 - phaseGouy + k * (X**2 + Y**2) /
                        (2 * R)))
    phase2 = np.exp(-1j * (k * z0 - phaseGouy + k * ((x0-X)**2 + (y0-Y)**2) /
                        (2 * R)))
    wavefront = amplitude * phase1 * phase2

    return wavefront 


def validate_beam_configuration(settings: BeamSettings):
    """Validate the user has passed the correct parameters for the given configuration"""
    # beam spread
    if settings.beam_spread is BeamSpread.Plane:

        if settings.source_distance is None:
            raise ValueError(f"A source_distance must be provided for {settings.beam_spread}")

        # plane wave is constant width along optical axis
        settings.final_diameter = settings.width
        logging.debug(f"The plane wave if constant along the optical axis. The beam final_diameter has been set to the initial width: {settings.width:.2e}m")

        if settings.distance_mode != DistanceMode.Direct:
            # plane waves only enabled for direct mode
            settings.distance_mode = DistanceMode.Direct
            logging.debug(f"Only DistanceMode.Direct is supported for BeamSpread.Plane. The distance_mode has been set to {settings.distance_mode}.")

    # can't do converging/divering square beams
    if settings.beam_spread in [BeamSpread.Converging, BeamSpread.Diverging]:
        settings.beam_shape = BeamShape.Circular
        logging.debug(f"Only BeamShape.Circular is supported for {settings.beam_spread}. The beam_shape has been set to {settings.beam_shape}.")

        # QUERY?
        if settings.theta == 0.0:
            if settings.numerical_aperture is None:
                raise ValueError(f"A non-zero theta or numerical aperture must be provided for a {settings.beam_spread} beam.")

    # beam shape
    # non-rectangular beams are symmetric
    if settings.beam_shape in [BeamShape.Circular]:
        settings.height = settings.width
        logging.debug(f"The beam_shape ({settings.beam_shape}) requires a symmetric beam. The beam height has been set to the beam width: {settings.width:.2e}m ")

    # distance mode
    if settings.distance_mode == DistanceMode.Direct:
        if settings.source_distance is None:
            raise ValueError(f"A source_distance must be provided for {settings.distance_mode}")

    if settings.distance_mode == DistanceMode.Focal:
        if settings.beam_spread not in [BeamSpread.Converging, BeamSpread.Diverging]:
            raise ValueError(f"BeamSpread must be Converging, or Diverging for {settings.distance_mode} (currently {settings.beam_spread})")

        if settings.focal_multiple is None:
            settings.focal_multiple = 1.0 # set default to 1.0

    if settings.distance_mode == DistanceMode.Diameter:
        if settings.final_diameter is None:
            raise ValueError(f"A final_diameter must be provided for {settings.distance_mode}")

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

def load_beam_config(config: dict) -> BeamSettings:
    """Load the beam settings from dictionary

    Args:
        config (dict): beam configuration as dictionary

    Returns:
        BeamSettings: beam configuration as BeamSettings
    """

    logging.warning(f"This function is deprecated and will be removed. Please use BeamSettings.__from_dict__ instead.")

    config = validation._validate_default_beam_config(config)

    beam_settings = BeamSettings(
        distance_mode=DistanceMode[config["distance_mode"]],
        beam_spread=BeamSpread[config["spread"]],
        beam_shape=BeamShape[config["shape"]],
        width=config["width"],
        height= config["height"],
        position_x=config["position_x"],
        position_y=config["position_y"],
        theta=config["theta"],
        numerical_aperture=config["numerical_aperture"],
        tilt_x=config["tilt_x"],
        tilt_y=config["tilt_y"],
        source_distance=config["source_distance"],
        final_diameter=config["final_diameter"],
        focal_multiple=config["focal_multiple"],
        n_steps=config["n_steps"],
        step_size=config["step_size"],
        output_medium=config["output_medium"],
        operator=BeamOperator[config["operator"]],
        gaussian_wx=config["gaussian_wx"],
        gaussian_wy=config["gaussian_wy"],
        gaussian_z0=config["gaussian_z0"],
        gaussian_z=config["gaussian_z"],
        data=config["data"]
    )

    return beam_settings

def generate_beam(config: dict, parameters: SimulationParameters):
    """Create a beam for simulation

    Args:
        config (dict): beam configuration
        parameters (SimulationParameters): global simulation parameters

    Returns:
        Beam: initial simulation beam
    """
    # load beam settings
    beam_settings = BeamSettings.__from__dict__(config)

    # create beam
    beam = Beam(settings=beam_settings)

    # generate profile
    beam.generate_profile(sim_parameters=parameters)

    # generate wavefront
    beam.generate_wavefront(parameters=parameters)

    return beam
