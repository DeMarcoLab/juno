from dataclasses import dataclass

import numpy as np

from pathlib import Path
from scipy import ndimage
from enum import Enum

from lens_simulation.Medium import Medium


@dataclass
class GratingSettings:
    width: float                # metres
    distance: float             # metres
    depth: float                # metres
    axis: int = 1
    centred: bool = True
    distance_px: int = None     # pixels
    width_px: int = None        # pixels

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
        self.aperture = None # area of lens that is apertured

    def __repr__(self):

        return f""" Lens (diameter: {self.diameter:.2e}, height: {self.height:.2e}, \nexponent: {self.exponent:.3f}, refractive_index: {self.medium.refractive_index:.3f}),"""

    def generate_profile(
        self, pixel_size: float, lens_type: LensType = LensType.Cylindrical, length: int = None
    ) -> np.ndarray:
        """_summary_

        Args:
            pixel_size (float): simulation pixel size
            lens_type (LensType, optional): method to create the lens profile [Cylindrical or Spherical]. Defaults to LensType.Cylindrical.
            length (int, optional): distance to extrude the profile. Defaults to None. (metres) 

        Returns:
            np.ndarray: lens profile
        """
        from lens_simulation.utils import _calculate_num_of_pixels

        n_pixels = _calculate_num_of_pixels(self.diameter, pixel_size, odd = True)

        self.pixel_size = pixel_size

        if lens_type == LensType.Cylindrical:
            
            if length is None:
                length = pixel_size
            
            self.profile = self.extrude_profile(length, n_pixels)

        if lens_type == LensType.Spherical:

            self.profile = self.revolve_profile(n_pixels)

        return self.profile


    def invert_profile(self):
        """Invert the lens profile"""
        if self.profile is None:
            raise RuntimeError(
                "This lens has no profile. Please generate the lens profile before inverting"
            )

        self.profile = abs(self.profile - np.max(self.profile))

        return self.profile

    def load_profile(self, fname: Path, pixel_size: float):
        """Load the lens profile from np array"""

        # load the profile
        arr = np.load(fname)

        self.pixel_size = pixel_size
        self.profile = arr

        return self.profile

    def extrude_profile(self, length: float, n_pixels: int) -> np.ndarray:
        """Extrude the lens profile to create a cylindrical lens profile.

        args:
            length: (int) the length of the extruded lens (metres)

        """
        from lens_simulation.utils import _calculate_num_of_pixels

        # generate 1d profile
        profile = create_profile_1d(self.diameter, self.height, self.exponent, n_pixels)

        # length in pixels
        length_px = _calculate_num_of_pixels(length, self.pixel_size, odd=True)

        # extrude profile
        profile = np.ones((length_px, *profile.shape)) * profile

        # filter profile
        # profile = ndimage.gaussian_filter(profile, sigma=3)

        return profile

    def revolve_profile(self, n_pixels: int):
        """Revolve the lens profile around the centre of the lens"""

        # len/sim parameters
        lens_width = self.diameter
        lens_length = self.diameter
        n_pixels_x = n_pixels  # odd
        n_pixels_y = n_pixels  # odd

        # revolve the profile around the centre (distance)
        x = np.linspace(0, lens_width, n_pixels_x)
        y = np.linspace(0, lens_length, n_pixels_y)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(((lens_width / 2) - X) ** 2 + ((lens_length / 2) - Y) ** 2)

        # general profile formula...
        # coefficient is defined on the radius, not diameter

        coefficient = self.height / (lens_width / 2) ** self.exponent
        profile = self.height - coefficient * distance ** self.exponent

        # clip the profile to zero
        profile = np.clip(profile, 0, np.max(profile))

        # filter profile
        # profile = ndimage.gaussian_filter(profile, sigma=3)

        return profile

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

    def apply_masks(
        self,
        grating: bool = False,
        truncation: bool = False,
        aperture: bool = False,
        escape_path: bool = True,
    ):

        # grating
        if grating:
            self.profile[self.grating_mask] -= self.grating_depth

        # truncation
        if truncation:
            self.profile[self.truncation_mask] = self.truncation

        # aperture
        if aperture:
            self.profile[self.custom_aperture_mask] = 0     # for profile visualisation
            self.aperture = self.custom_aperture_mask       # for propagation

        # if escape_path:
        #     self.profile = self.calculate_escape_path(ratio=0.2)

        # clip the profile to zero
        self.profile = np.clip(self.profile, 0, np.max(self.profile))

        return self.profile

    def calculate_truncation_mask(
        self, truncation: float = None, radius: float = 0, type: str = "value"
    ):
        """Calculate the truncation mask """
        if self.profile is None:
            raise RuntimeError(
                "A Lens profile must be defined before applying a mask. Please generate a profile first."
            )

        # if radius == 0:
        #     self.truncation_mask = np.zeros_like(self.profile).astype(int)
        #     self.truncation = np.max(self.profile)
        #     return self.profile

        if type == "value":

            if truncation is None:
                raise ValueError(
                    f"No truncation value has been supplied for a {type} type truncation. Please provide a maximum truncation height"
                )

        if type == "radial":
            radius_px = int(radius / self.pixel_size)

            truncation_px = self.profile.shape[0] // 2 + radius_px

            truncation = self.profile[self.profile.shape[0] // 2, truncation_px]

        self.truncation_mask = self.profile >= truncation
        self.truncation = truncation

        return self.profile

    def calculate_escape_path(self, ratio=0.2):

        inner_px = int(self.profile.shape[1] / self.pixel_size)
        outer_px = int(self.profile.shape[1] * (1 + 0.2) / self.pixel_size)

        mask = np.ones_like(self.profile)

        if type == "radial":

            inner_rad_px = self.profile.shape[1] // 2 + inner_px
            outer_rad_px = self.profile.shape[1] // 2 + outer_px

            inner_val = self.profile[self.profile.shape[0] // 2, inner_rad_px]
            outer_val = self.profile[self.profile.shape[0] // 2, outer_rad_px]

            inner_mask = self.profile <= inner_val
            outer_mask = self.profile >= outer_val
            mask[inner_mask * outer_mask] = 0

        h, w = self.profile.shape

        h1, w1 = int(h * (1 + ratio)), int(w * (1 + ratio))
        profile_with_escape_path = np.zeros(shape=(h1, w1))

        profile_with_escape_path[:h, :w] = self.profile

        self.profile = profile_with_escape_path

        return NotImplemented

    # escape path:
    # inner radius: lens radius
    # outer radius: (1 + scale) * lens radius
    # e.g. 1.1 * lens_radius
    # pad with zero?

    def calculate_aperture(
        self,
        inner_m: float = 0,
        outer_m: float = 0,
        type: str = "square",
        inverted: bool = False,
    ):
        """Calculate the aperture mask"""

        # Aperture
        # define mask, apply to profile (wavefront = 0)

        if inner_m > outer_m:
            raise ValueError(
                """Inner radius must be smaller than outer radius.
                            inner = {inner_m:.1e}m, outer = {outer_m:.1e}m"""
            )

        if inner_m == 0.0 and outer_m == 0.0:
            self.custom_aperture_mask = np.zeros_like(self.profile, dtype=np.uint8)
            return self.profile

        inner_px = int(inner_m / self.pixel_size)
        outer_px = int(outer_m / self.pixel_size)

        if not inverted:
            mask = np.zeros_like(self.profile, dtype=np.uint8)
        else:
            mask = np.ones_like(self.profile, dtype=np.uint8)

        centre_x_px = mask.shape[1] // 2
        centre_y_px = mask.shape[0] // 2

        if type == "square":

            min_x, max_x = centre_x_px - inner_px, centre_x_px + inner_px
            min_y, max_y = centre_y_px - inner_px, centre_y_px + inner_px

            outer_x0, outer_x1 = centre_x_px - outer_px, centre_x_px + outer_px
            outer_y0, outer_y1 = centre_y_px - outer_px, centre_y_px + outer_px

            mask[outer_y0:outer_y1, outer_x0:outer_x1] = 1
            mask[min_y:max_y, min_x:max_x] = 0

        if type == "radial":

            inner_rad_px = self.profile.shape[1] // 2 + inner_px
            outer_rad_px = self.profile.shape[1] // 2 + outer_px

            inner_val = self.profile[self.profile.shape[0] // 2, inner_rad_px]
            outer_val = self.profile[self.profile.shape[0] // 2, outer_rad_px]

            inner_mask = self.profile <= inner_val
            outer_mask = self.profile >= outer_val

            mask[inner_mask * outer_mask] = not inverted

        self.custom_aperture_mask = mask == 1

        return self.profile

    def calculate_grating_mask(
        self,
        settings: GratingSettings,
        x_axis: bool = True,
        y_axis: bool = False,
    ) -> None:
        """Calculate the grating mask for the specified settings"""

        if settings.width == 0.0:
            self.grating_mask = np.zeros_like(self.profile, dtype=np.uint8)
            self.grating_depth = 0
            return self.profile

        if settings.width >= settings.distance:
            raise ValueError(
                f"""Grating width cannot be equal or larger than the distance between gratings.
                                    width={settings.width:.2e}, distance = {settings.distance:.2e}"""
            )
        
        # cannot apply gratings to 1d array
        if self.profile.shape[0] == 1:
            y_axis = False

        settings.width_px = int(settings.width / self.pixel_size)
        settings.distance_px = int(settings.distance / self.pixel_size)

        grating_coords_x = calculate_grating_coords(self.profile, settings, axis=1)
        grating_coords_y = calculate_grating_coords(self.profile, settings, axis=0)

        mask = np.zeros_like(self.profile, dtype=np.uint8)

        if x_axis:
            mask[:, grating_coords_x] = 1

        if y_axis:
            mask[grating_coords_y, :] = 1

        self.grating_mask = mask == 1
        self.grating_depth = settings.depth






def create_profile_1d(diameter: float, height: float, exponent: float, n_pixels: int) -> np.ndarray:
    """Create 1 dimensional lens profile"""
    
    # # make functional
    # height = self.height
    # exponent = self.exponent

    radius = diameter / 2
    n_pixels_in_radius = n_pixels // 2 + 1

    # x coordinate of pixels
    radius_px = np.linspace(0, radius, n_pixels_in_radius)

    # determine coefficent at boundary conditions
    # TODO: will be different for Hershel, Paraxial approximation)
    coefficient = height / (radius ** exponent)

    # generic lens formula
    # H = h - C*r ^ e
    heights = height - coefficient * radius_px ** exponent

    # generate symmetric height profile (NOTE: assumed symmetric lens).
    profile = np.append(np.flip(heights[1:]), heights)

    # always smooth
    profile = ndimage.gaussian_filter(profile, sigma=3)

    return profile

def calculate_grating_coords(profile: np.ndarray, settings: GratingSettings, axis: int = 1,
):

    """Calculate the positions of grating coordinates for the specified axis"""

    start_coord = profile.shape[axis] // 2

    if settings.centred:
        # start at centre pixel then move out
        start_coord_1 = profile.shape[axis] // 2
        start_coord_2 = profile.shape[axis] // 2

    else:
        # start at centre pixel + half then move out
        start_coord_1 = start_coord - settings.distance_px // 2
        start_coord_2 = start_coord + settings.distance_px // 2

    # coordinates of grating centres (starting from centre to both edges)
    grating_centre_coords_x1 = np.arange(start_coord_1, 0, -settings.distance_px)
    grating_centre_coords_x2 = np.arange(start_coord_2, profile.shape[axis], settings.distance_px)
    grating_centre_coords = sorted(np.append(grating_centre_coords_x1, grating_centre_coords_x2))

    # coordinates of full grating which is grating centre +/- width / 2
    grating_coords = []

    for px in grating_centre_coords:

        min_px, max_px = px - settings.width_px / 2, px + settings.width_px / 2

        grating = np.arange(min_px, max_px).astype(int)

        grating = np.clip(grating, 0, profile.shape[axis] - 1)

        grating_coords.append(grating)

    return np.ravel(grating_coords)


def apply_modifications(lens: Lens, lens_config: dict) -> Lens:
    """Apply all lens modifications defined in config"""
    if lens_config["grating"] is not None:
        grating_settings = GratingSettings(
                width=lens_config["grating"]["width"],
                distance=lens_config["grating"]["distance"],
                depth=lens_config["grating"]["depth"],
                centred=lens_config["grating"]["centred"],
            )
        lens.calculate_grating_mask(
                grating_settings,
                x_axis=lens_config["grating"]["x"],
                y_axis=lens_config["grating"]["y"],
            )

    if lens_config["truncation"] is not None:
        lens.calculate_truncation_mask(
                truncation=lens_config["truncation"]["height"],
                radius=lens_config["truncation"]["radius"],
                type=lens_config["truncation"]["type"],
            )

    if lens_config["aperture"] is not None:
        lens.calculate_aperture(
                inner_m=lens_config["aperture"]["inner"],
                outer_m=lens_config["aperture"]["outer"],
                type=lens_config["aperture"]["type"],
                inverted=lens_config["aperture"]["invert"],
            )

        # apply masks
    use_grating = True if lens_config["grating"] is not None else False
    use_truncation = (True if lens_config["truncation"] is not None else False)
    use_aperture = True if lens_config["aperture"] is not None else False

    lens.apply_masks(
            grating=use_grating,
            truncation=use_truncation,
            aperture=use_aperture,
        )
        
    return lens