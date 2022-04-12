from dataclasses import dataclass
import numpy as np

from scipy import ndimage

# TODO: 488 comes from sim
# TODO: fix the __repr__ for Medium
# TODO: comparison not working for dataclass?
@dataclass
class Medium:
    def __init__(self, refractive_index: float = 1.0) -> None:
        self.refractive_index = refractive_index
        self.wavelength_medium: float = 488e-9 / self.refractive_index
        self.wave_number: float = 2 * np.pi / self.wavelength_medium


@dataclass
class Water(Medium):
    refractive_index: float = 1.33



# TODO:
# - grating

class Lens:
    def __init__(
        self, diameter: float, height: float, exponent: float, medium: Medium = Medium()
    ) -> None:

        self.diameter = diameter
        self.height = height
        self.exponent = exponent
        self.medium = medium
        self.escape_path = None
        self.profile = None

    def __repr__(self):

        return f""" Lens (diameter: {self.diameter:.2e}, height: {self.height:.2f}, \nexponent: {self.exponent:.3f}, refractive_index: {self.medium.refractive_index:.3f}),"""

    def generate_profile(self, pixel_size) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        # TODO: someone might define using the n_pixels

        radius = self.diameter / 2
        n_pixels = int(radius / pixel_size)
        # n_pixels must be odd (symmetry).
        if n_pixels % 2 == 0:
            n_pixels += 1

        # x coordinate of pixels (TODO: better name)
        radius_px = np.linspace(0, radius, n_pixels)

        # determine coefficent at boundary conditions
        # TODO: will be different for Hershel, Paraxial approximation)
        coefficient = self.height / (radius**self.exponent)

        # generic lens formula
        # H = h - C*r ^ e
        heights = self.height - coefficient * radius_px**self.exponent

        # generate symmetric height profile (NOTE: assumed symmetric lens).
        profile = np.append(np.flip(heights[1:]), heights)

        # always smooth
        profile = ndimage.gaussian_filter(profile, sigma=3)

        self.profile = profile

        return profile
    
    def invert_profile(self):
        """Invert the lens profile"""
        if self.profile is None:
            raise RuntimeError("This lens has no profile. Please generate the lens profile before inverting")
     
        self.profile = abs(self.profile - np.max(self.profile))

        return self.profile


    def load_profile(self, arr: np.ndarray, pixel_size: int):
        """Load the lens profile from np array"""

        self.profile = arr

        # TODO: check the profile is the required size/shape
        # assert len(self.profile) == 

        return self.profile
