from dataclasses import dataclass
from math import trunc
import numpy as np

from scipy import ndimage
from enum import Enum

from lens_simulation.Medium import Medium

# TODO:
# - grating


class LensType(Enum):
    Cylindrical = 1
    Spherical = 2


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

    def generate_profile(self, pixel_size, lens_type: LensType = LensType.Cylindrical) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        # TODO: someone might define using the n_pixels

        radius = self.diameter / 2
        n_pixels = int(radius / pixel_size) # n_pixels in radius
        # n_pixels must be odd (symmetry). 
        if n_pixels % 2 == 0:
            n_pixels += 1

        self.pixel_size = pixel_size
        self.n_pixels = n_pixels
        self.lens_type = lens_type

        if self.lens_type == LensType.Cylindrical:
        
            self.profile = self.create_profile_1d(radius, n_pixels)
        
        if self.lens_type == LensType.Spherical:

            self.profile = self.revolve_profile()

        return self.profile

    def create_profile_1d(self, radius, n_pixels):

        # x coordinate of pixels
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
            raise ValueError(
                f"Custom lens profiles must match the simulation width. Custom Profile Shape: {arr.shape}, Simulation Pixels: {self.n_pixels}."
            )

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

        # regenerate the profile
        self.generate_profile(self.pixel_size)

        # length in pixels
        length_px = int(length // self.pixel_size)

        # extrude profile
        self.profile = np.ones((length_px, *self.profile.shape)) * self.profile

        return self.profile

    def revolve_profile(self):
        """Revolve the lens profile around the centre of the lens"""

        # len/sim parameters
        lens_width = self.diameter 
        lens_length = self.diameter
        n_pixels_x = self.n_pixels * 2
        n_pixels_y = self.n_pixels * 2
                
        # revolve the profile around the centre (distance)
        x = np.linspace(0, lens_width, n_pixels_x)
        y = np.linspace(0, lens_length, n_pixels_y)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(((lens_width / 2) - X) ** 2 + ((lens_length / 2) - Y) ** 2)

        # general profile formula...
        # coefficient is defined on the radius, not diameter

        coefficient = self.height / (lens_width/2) ** self.exponent
        profile = self.height - coefficient * distance ** self.exponent

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
    """

    def calculate_truncation_mask(self, truncation: float = None, radius: float = 0, type: str = "value"):
        """Calculate the truncation mask """
        if self.profile is None:
            raise RuntimeError("A Lens profile must be defined before applying a mask. Please generate a profile first.")

        if radius == 0:
            # self.truncation_mask = np.zeros_like(self.profile)
            # self.truncation = np.max(self.profile)
            return self.profile

        if type == "value":

           if truncation is None:
                raise ValueError(f"No truncation value has been supplied for a {type} type truncation. Please provide a maximum truncation height")

        if type == "radial":
            radius_px = int(radius/self.pixel_size)

            truncation_px = self.profile.shape[0] // 2 + radius_px

            truncation = self.profile[self.profile.shape[0] // 2, truncation_px]

        self.truncation_mask = self.profile >= truncation
        self.truncation = truncation
        
        return self.profile

    def apply_masks(self, grating: bool = False, truncation: bool = False, apeture: bool = False):
        
        # grating
        if grating:
            self.profile[:, self.grating_mask] -= self.grating_depth

        # truncation
        if truncation:
            self.profile[self.truncation_mask] = self.truncation

        # apeture
        if apeture:
            self.profile = self.profile * self.apeture_mask

        # clip the profile to zero
        self.profile = np.clip(self.profile, 0, np.max(self.profile))

        return self.profile


    def calculate_apeture(self, inner_m: float = 0, outer_m: float = 0, type: str = "square"):
        """Calculate the apeture mask"""

        # Apeture
        # define mask, apply to profile (wavefront = 0)

        if inner_m > outer_m:
            raise ValueError("""Inner radius must be smaller than outer radius. 
                            inner = {inner_m:.1e}m, outer = {outer_m:.1e}m""")

        if inner_m == 0.0 and outer_m == 0.0:
            self.apeture_mask = np.ones_like(self.profile)
            # self.apeture_value = 0
            return self.profile

        inner_px = int(inner_m/self.pixel_size)
        outer_px = int(outer_m/self.pixel_size) 

        mask = np.ones_like(self.profile)

        centre_x_px = mask.shape[1] // 2
        centre_y_px = mask.shape[0] // 2

        if type=="square":

            min_x, max_x = centre_x_px - inner_px, centre_x_px + inner_px
            min_y, max_y = centre_y_px - inner_px, centre_y_px + inner_px 

            outer_x0, outer_x1 = centre_x_px - outer_px, centre_x_px + outer_px
            outer_y0, outer_y1 = centre_y_px - outer_px, centre_y_px + outer_px

            mask[outer_y0: outer_y1, outer_x0: outer_x1] = 0
            mask[min_y:max_y, min_x:max_x  ] = 1
        
        if type=="radial":

            inner_rad_px = self.profile.shape[1] // 2 + inner_px
            outer_rad_px = self.profile.shape[1] // 2 + outer_px

            inner_val = self.profile[self.profile.shape[0] // 2, inner_rad_px]
            outer_val = self.profile[self.profile.shape[0] // 2, outer_rad_px]

            inner_mask = self.profile <= inner_val
            outer_mask = self.profile >= outer_val
            mask[inner_mask * outer_mask] = 0

        self.apeture_mask = mask

        return self.profile



    def calculate_grating_mask(self, grating_width_m: float, distance_m: float, depth_m: float, centred: bool = True):
        
        if grating_width_m >= distance_m:
            raise ValueError(f"""Grating width cannot be equal or larger than the distance between gratings. 
                                    width={grating_width_m:.2e}, distance = {distance_m:.2e}""")

        if grating_width_m == 0.0:
            
            return self.profile 
        
        # TODO: gratings in y directions

        grating_width_px = int(grating_width_m / self.pixel_size)
        distance_px = int(distance_m / self.pixel_size)
        grating_centre_coords_x = np.arange(0, self.profile.shape[1], distance_px)

        # this coord +/- width / 2
        grating_coords_x = []

        for px in grating_centre_coords_x:

            min_px, max_px = px - grating_width_px / 2, px + grating_width_px / 2

            grating_x = np.arange(min_px, max_px).astype(int)

            grating_x = np.clip(grating_x, 0, self.profile.shape[1]-1)
            
            grating_coords_x.append(grating_x)

        self.grating_mask = np.ravel(grating_coords_x)
        self.grating_depth = depth_m

        return self.profile


    # def calculate_grating_parameters(self):
    #     # grating calculations
    #     if self.grating:
    #         buffer = 0
    #         if self.grating_edge:
    #             buffer = 1
    #         self.grating_points = list()

    #         if self.grating_mode == 'distance':
    #             # TODO: only define grating by width, not px
    #             self.grating_spacing_px = int(self.grating_spacing/self.pixel_size_x)
    #             if self.grating_centered:
    #                 point = self.center_x_px
    #             else:
    #                 point = self.center_x_px - self.grating_spacing_px/2

    #             while point < self.center_x_px + self.radius_px + buffer:
    #                 self.grating_points.append(point)
    #                 point += self.grating_spacing_px

    #             if self.grating_centered:
    #                 point = self.center_x_px - self.grating_spacing_px
    #             else:
    #                 point = self.center_x_px - self.grating_spacing_px*3/2

    #             while point > self.center_x_px - self.radius_px - buffer:
    #                 self.grating_points.append(point)
    #                 point -= self.grating_spacing_px

    #             # TODO: maybe remove this?
    #             self.grating_count = len(self.grating_points)

    #         # TODO: Add more Y things eventually
    #         elif self.grating_mode == 'count':
    #             step_size = self.diameter/(self.grating_count+1)
    #             step_size_px = step_size/self.pixel_size_x
    #             for i in range(self.grating_count):
    #                 self.grating_points.append(self.center_x_px-self.radius_px
    #                                            + (i + 1) * step_size_px)
    #             if self.grating_edge:
    #                 self.grating_points.append(self.center_x_px-self.radius_px)
    #                 self.grating_points.append(self.center_x_px+self.radius_px)

    #             self.grating_spacing = self.diameter / self.grating_count
    #             self.grating_spacing_px = int(
    #                 self.n_pixels_x / self.sim_width * self.grating_width)
    #         else:
    #             raise ValueError('Grating type not accepted')

    #         if self.dimension == 1:
    #             self.grating_direction = 'x'
    #         if self.grating_direction == 'x':
    #             grating_width_px_half = int((self.grating_width/self.pixel_size_x)/2)
    #         else:
    #             grating_width_px_half = int((self.grating_width/self.pixel_size_y)/2)
    #         for position in self.grating_points:
    #             start = int(np.floor(position - grating_width_px_half))
    #             end = int(np.ceil(position + grating_width_px_half))
    #             if self.grating_direction == 'x':
    #                 self.grating_mask[start:end + 1] = self.grating_depth
    #             else:
    #                 self.grating_mask[:, start:end + 1] = self.grating_depth