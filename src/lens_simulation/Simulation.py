from dis import dis
import numpy as np
from scipy import fftpack
import os
from lens_simulation.Lens import Lens, Medium
import uuid
from typing import Optional

import petname


class Simulation:
    def __init__(self, config: dict) -> None:

        self.sim_id = uuid.uuid4()
        self.petname = petname.Generate(2)  # TODO: maybe
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
        print("-" * 50)
        print(f"Running Simulation {self.petname} ({str(self.sim_id)[-10:]})")
        print(f"Parameters:  {self.parameters}")

        # TODO: actually run the simulation
        # generate_lens_profile
        # generate_medium_mesh
        # generate_frequency_array
        # generate_differential_refractive_profile
        # internal_lens_propagation
        # free_space_propagation

        print("-" * 50)


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


def generate_differential_refractive_index_profile(
    lens: Lens,
    z_resolution: int,
    previous_slice: np.ndarray,
    output_medium: Medium,
    pixel_size: float,
    rounding: Optional[int] = 0,
) -> np.ndarray:
    """Creates an n+1 dimensional array from an n dimensional lens to describe
    the discretised changes in refractive index along the lens profile.

    Parameters
    ----------
    lens : Lens
        lens to create the differential phase map of
    z_resolution : int
        step size of the discretisation of the lens
    previous_slice : np.ndarray
        an ndarray of the refractive index values of the previous medium
    output_medium : Medium
        the medium to which light will propagate after the lens
    pixel_size : float
        the pixel_size of the lens
    rounding : int, optional
        whether or not to increase the rounding of the discretisation.
        (0 is binary, 1 is to the nearest 0.1)

    Returns
    -------
    np.ndarray
        discretised n+1 dimensional array of refractive index changes
    """

    if lens.profile is None:
        lens.generate_profile(pixel_size)

    # generate the discretised profile
    discrete_profile = generate_discrete_profile(
        lens=lens,
        z_resolution=z_resolution,
        output_medium=output_medium,
        rounding=rounding,
    )

    # set lens values to the refractive index
    discrete_profile[discrete_profile != 0] = lens.medium.refractive_index * discrete_profile[discrete_profile != 0] / z_resolution

    # set the non-lens pixel values to the output medium
    discrete_profile[discrete_profile == 0] = output_medium.refractive_index
    discrete_profile[-1] = output_medium.refractive_index

    # create empty refractive index profile array
    refractive_index_differential_profile = np.zeros(shape=discrete_profile.shape)

    # calculate refractive differential profile
    refractive_index_differential_profile[1:] = discrete_profile[:-1] - discrete_profile[1:]
    refractive_index_differential_profile[0] = previous_slice - discrete_profile[0]

    return refractive_index_differential_profile


def generate_discrete_profile(
    lens: Lens, z_resolution: int, output_medium: Medium, rounding: Optional[int] = 0
):
    """_summary_

    Parameters
    ----------
    lens : Lens
        lens to create the discrete profile of
    z_resolution : int
        step size of the discretisation of the lens
    output_medium : Medium
        the medium to which light will propagate after the lens
    rounding : int, optional
        whether or not to increase the rounding of the discretisation.
        (0 is binary, 1 is to the nearest 0.1)

    Returns
    -------
    np.ndarray
        discretised n+1 dimensional array of the profile
    """

    # calculate the number of layers to split the lens into
    n_steps = int(np.ceil(max(lens.profile) / z_resolution))
    print(n_steps)
    # set up empty arrays for new profile
    discrete_profile = np.zeros(shape=(n_steps + 1, len(lens.profile)))

    for step in range(n_steps):
        above = lens.profile >= z_resolution * step
        new_profile = lens.profile * above - z_resolution * step
        new_profile[new_profile < 0] = 0
        new_profile[new_profile >= z_resolution] = z_resolution
        discrete_profile[step] = new_profile

    # # interpolate the values of the lens between 0 and n_steps
    # interpolated_profile = (lens.profile / max(lens.profile)) * n_steps

    # # round each pixel of the profile to the nearest decimal 'rounding' value
    # for i, pixel in enumerate(interpolated_profile):
    #     interpolated_profile[i] = (
    #         round(interpolated_profile[i], rounding) * z_resolution
    #     )

    # discretise into n+1 dimensional array
    # for step in range(n_steps):
    #     # create mask where values above current step exist
    #     discrete_profile[step] = lens.profile >= z_resolution * step

    return discrete_profile
