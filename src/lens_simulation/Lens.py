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


@dataclass
class Air(Medium):
    refactive_index: float = 1.00


@dataclass
class LithiumNiabate(Medium):
    refractive_index: float = 2.348


# TODO: add more common mediums.. (and names)


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

        return f""" Lens (diameter: {self.diameter:.2e}, height: {self.height:.2e}, \nexponent: {self.exponent:.3f}, refractive_index: {self.medium.refractive_index:.3f}),"""

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

        self.pixel_size = pixel_size
        self.n_pixels = n_pixels

        # x coordinate of pixels (TODO: better name)
        radius_px = np.linspace(0, radius, n_pixels)
        self.radius_px = radius_px

        # determine coefficent at boundary conditions
        # TODO: will be different for Hershel, Paraxial approximation)
        coefficient = self.height / (radius ** self.exponent)

        # generic lens formula
        # H = h - C*r ^ e
        heights = self.height - coefficient * radius_px ** self.exponent

        # generate symmetric height profile (NOTE: assumed symmetric lens).
        profile = np.append(np.flip(heights[1:]), heights)

        # always smooth
        profile = ndimage.gaussian_filter(profile, sigma=3)

        self.profile = profile

        return profile

    def invert_profile(self):
        """Invert the lens profile"""
        if self.profile is None:
            raise RuntimeError(
                "This lens has no profile. Please generate the lens profile before inverting"
            )

        self.profile = abs(self.profile - np.max(self.profile))

        return self.profile

    def load_profile(self, arr: np.ndarray):
        """Load the lens profile from np array"""

        # assume lens diameter is sim width
        if arr.shape[-1] != self.n_pixels:
            raise ValueError(f"Custom lens profiles must match the simulation width. Custom Profile Shape: {arr.shape}, Simulation Pixels: {self.n_pixels}.")

        # TODO: we need pad the lens if the size is smaller than the sim n_pixels?

        self.profile = arr
        
        return self.profile

    def extrude_profile(self, length: float) -> np.ndarray:
        """Extrude the lens profile to create a cylindrical lens profile.
        
        args:
            length: (int) the length of the extruded lens (metres)
        
        """
        if self.profile is None:
            raise RuntimeError(
                "This lens has no profile. Please generate the lens profile before extruding"
            )

        # TODO: should probably regenerate the profile here to be certain
        self.generate_profile(self.pixel_size)

        # length in pixels
        length_px = int(length // self.pixel_size)

        # extrude profile       
        self.profile = np.ones((length_px, *self.profile.shape)) * self.profile
                
        return self.profile


    def revolve_profile(self):
        """Revolve the lens profile around the centre of the lens"""

        # TODO: remove / refactor
        self.sim_width = self.diameter 
        self.sim_height = self.diameter
        self.n_pixels_x = self.n_pixels
        self.n_pixels_y = self.n_pixels


        # TODO: validate what sim_width, sim_height represent
        # TODO: validate the dimensions of this, doesn't seem correct (seems to be half sized)


        x = np.linspace(0, self.sim_width, self.n_pixels_x)
        y = np.linspace(0, self.sim_height, self.n_pixels_y)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(((self.sim_width/2)-X)**2 + ((self.sim_height/2)-Y)**2)
        self.angle = np.arctan2(self.sim_width/2-X, self.sim_height/2-Y + 1e-12)
       

        # QUERY: do we need special axicon case?

        # general profile formula...
        coefficient = self.height / max(self.radius_px ** self.exponent)
        profile = self.height - coefficient * distance ** self.exponent
        
        # QUERY: we dont need padding here?
        # profile = np.pad(profile, (int((self.n_pixels_x-len(profile))/2), int((self.n_pixels_x-len(profile))/2)), 'constant')

        # clip the profile to zero
        profile = np.clip(profile, 0, np.max(profile))

        # override 1D profile
        self.profile = profile

        return self.profile

    """
    x: zero, equal
    M: max
    #####################
    #         x         #
    #                   #
    #                   #
    #x        M        x#
    #                   #
    #                   #
    #         x         #
    #####################

P

    """
        

# TODO:
# top down "heatmap/contour" for 2D lens