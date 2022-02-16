import numpy as np
from scipy import fftpack
import os
from lens_simulation.Lens import Lens, Medium
import uuid

import petname

class Simulation:

    def __init__(self, config: dict) -> None:

        self.sim_id = uuid.uuid4()
        self.petname = petname.Generate(2) # TODO: maybe
        self.read_configuration(config=config)
        self.setup_simulation()

    def read_configuration(self, config):

        # TODO: add a check to the config
        self.config = config
        self.run_id = config["run_id"]
        self.parameters = config["parameters"]

    def setup_simulation(self):

        self.log_dir = os.path.join(self.config["log_dir"], str(self.sim_id))
        os.makedirs(self.log_dir, exist_ok=True)

    def run_simulation(self):
        print("-"*50)
        print(f"Running Simulation {self.petname} ({str(self.sim_id)[-10:]})")
        print(f"Parameters:  {self.parameters}")

        # TODO: actually run the simulation
        # generate_lens_profile
        # generate_medium_mesh
        # generate_frequency_array
        # generate_differential_refractive_profile
        # internal_lens_propagation
        # free_space_propagation

        print("-"*50)






def generate_squared_frequency_array(n_pixels: int, pixel_size: float) -> np.ndarray:
    """Generates the squared frequency array used in the fresnel diffraction integral

    Parameters
    ----------
    n_pixels : int
        number of pixels in the lens array
    pixel_size : float
        realspace size of each pixel in the lens array

    Returns
    -------
    np.ndarray
        squared frequency array
    """
    return np.power(fftpack.fftfreq(n_pixels, pixel_size), 2)


def calculate_equivalent_focal_distance(lens: Lens, medium: Medium) -> float:
    """Calculates the focal distance of a lens with any exponent as if it had
    an exponent of 2.0 (assuming plano-concave focusing lens)

    Parameters
    ----------
    lens : Lens
        Lens instance as defined in Lens.py
    medium : Medium
        Medium instance as defined in Lens.py

    Returns
    -------
    float
        returns the calculated focal distance
    """
    equivalent_radius_of_curvature = 0.5 * (
        lens.height + (lens.diameter / 2) ** 2 / lens.height
    )
    equivalent_focal_distance = (
        equivalent_radius_of_curvature * medium.refractive_index
    ) / (lens.medium.refractive_index - medium.refractive_index)

    return equivalent_focal_distance


def generate_discrete_profile(
    profile: np.ndarray, z_resolution: int, rounding: int
) -> np.ndarray:
    """Creates an n+1 dimensional array from an n dimensional lens to split the
    height map into discrete blocks to be simulated separately

    Parameters
    ----------
    profile : np.ndarray
        original n-dimensional profile to be discretised
    z_resolution : int
        step size in the direction of light propagation
    rounding : int
        the integer digit to round to (0 is binary)

    Returns
    -------
    np.ndarray
        discretised n+1 dimensional array
    """

    # calculate the number of layers to split the lens into
    n_steps = int(np.ceil(max(profile)/z_resolution))

    # empty array for new profile
    discrete_profile = np.zeros(shape=(n_steps, len(profile)))

    # interpolate the values of the lens between 0 and n_steps
    interpolated_profile = (profile / max(profile)) * n_steps

    for i, pixel in enumerate(interpolated_profile):
        interpolated_profile[i] = round(interpolated_profile[i], rounding) * z_resolution

    for step in range(n_steps):
        values_above_step = interpolated_profile > z_resolution * step
        discrete_profile[step] = values_above_step * z_resolution

    return discrete_profile
