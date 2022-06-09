from dataclasses import dataclass

import numpy as np

from pathlib import Path
from scipy import ndimage
from enum import Enum

from lens_simulation.Medium import Medium
from lens_simulation import validation


@dataclass
class GratingSettings:
    width: float  # metres
    distance: float  # metres
    depth: float  # metres
    axis: int = 1
    centred: bool = True
    distance_px: int = None  # pixels
    width_px: int = None  # pixels


class LensType(Enum):
    Cylindrical = 1
    Spherical = 2


class Lens:
    def __init__(
        self, diameter: float, height: float, exponent: float, medium: Medium = Medium(), lens_type: LensType = LensType.Spherical
    ) -> None:

        # lens properties
        self.diameter = diameter
        self.height = height
        self.exponent = exponent
        self.medium = medium
        self.length = diameter
        self.lens_type = lens_type

        # lens profile
        self.profile = None

        # aperturing masks
        self.non_lens_mask = None  # non defined lens area
        self.custom_aperture_mask = None  # user defined aperture
        self.truncation_aperture_mask = None  # truncation aperture
        self.sim_aperture_mask = None  # simulation padding aperture
        self.aperture = None  # full aperture mask

        # modifications
        self.grating_mask = None
        self.escape_mask = None
        self.truncation_mask = None

    def __repr__(self):

        return f""" Lens (diameter: {self.diameter:.2e}, height: {self.height:.2e}, \nexponent: {self.exponent:.3f}, refractive_index: {self.medium.refractive_index:.3f}),"""

    def generate_profile(
        self,
        pixel_size: float,
        length: int = None,
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

        n_pixels = _calculate_num_of_pixels(self.diameter, pixel_size, odd=True)

        self.pixel_size = pixel_size

        if self.lens_type == LensType.Cylindrical:

            if length is None:
                length = pixel_size
            self.length = length

            self.profile = self.extrude_profile(length, n_pixels)

        if self.lens_type == LensType.Spherical:

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

        # mask out non defined lens area
        self.non_lens_mask = (profile == 0).astype(
            bool
        )  # TODO: maybe change to distance filter instead?

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
        self, grating: bool = False, truncation: bool = False, aperture: bool = False,
    ):

        # grating
        if grating:
            self.profile[self.grating_mask] -= self.grating_depth

        # truncation
        if truncation:
            self.profile[self.truncation_mask] = self.truncation_height

        # aperture
        if aperture:
            self.apply_aperture_masks()

        # clip the profile to zero
        self.profile = np.clip(self.profile, 0, np.max(self.profile))

        return self.profile

    def create_truncation_mask(
        self,
        truncation_height: float = 0.0,
        radius: float = 0,
        type: str = "value",
        aperture: bool = False,
    ):
        """Create the truncation mask

        Args:
            height (float, optional): _description_. Defaults to None.
            radius (float, optional): _description_. Defaults to 0.
            type (str, optional): _description_. Defaults to "value".
            aperture (bool, optional): _description_. Defaults to False.

        Raises:
            RuntimeError: lens profile must be generated before truncating

        Returns:
            _type_: _description_
        """
        if self.profile is None:
            raise RuntimeError(
                "A Lens profile must be defined before applying a mask. Please generate a profile first."
            )

        if type == "radial":
            radius_px = int(radius / self.pixel_size)

            truncation_px = self.profile.shape[0] // 2 + radius_px

            truncation_height = self.profile[self.profile.shape[0] // 2, truncation_px]

        self.truncation_mask = self.profile >= truncation_height
        self.truncation_height = truncation_height

        if aperture:
            self.truncation_aperture_mask = self.truncation_mask

        return self.profile

    def create_custom_aperture(
        self,
        inner_m: float = 0,
        outer_m: float = 0,
        type: str = "square",
        inverted: bool = False,
    ):
        """Calculate the aperture mask"""
        from lens_simulation import utils

        if inner_m > outer_m:
            raise ValueError(
                """Inner dimension must be smaller than outer dimension.
                            inner = {inner_m:.1e}m, outer = {outer_m:.1e}m"""
            )

        # no aperture
        if inner_m == 0.0 and outer_m == 0.0:
            self.custom_aperture_mask = np.zeros_like(self.profile, dtype=bool)
            return

        h, w = self.profile.shape
        inner_px = int(inner_m / self.pixel_size)
        outer_px = int(outer_m / self.pixel_size)

        if type == "square":

            mask = np.zeros_like(self.profile, dtype=bool)

            centre_y_px = mask.shape[0] // 2
            centre_x_px = mask.shape[1] // 2

            # need to clip to make sure inside profile

            # inside square
            min_x = np.clip(centre_x_px - inner_px, 0, mask.shape[1])
            max_x = np.clip(centre_x_px + inner_px, 0, mask.shape[1])
            min_y = np.clip(centre_y_px - inner_px, 0, mask.shape[0])
            max_y = np.clip(centre_y_px + inner_px, 0, mask.shape[0])

            # outside square
            outer_x0 = np.clip(centre_x_px - outer_px, 0, mask.shape[1])
            outer_x1 = np.clip(centre_x_px + outer_px, 0, mask.shape[1])
            outer_y0 = np.clip(centre_y_px - outer_px, 0, mask.shape[0])
            outer_y1 = np.clip(centre_y_px + outer_px, 0, mask.shape[0])

            mask[outer_y0:outer_y1, outer_x0:outer_x1] = True
            mask[min_y:max_y, min_x:max_x] = False

        if type == "radial":

            # TODO: only works for spherical?

            distance = utils.create_distance_map_px(w, h)
            mask = ((distance <= outer_px) * (distance >= inner_px)).astype(bool)

        if inverted:
            mask = np.logical_not(mask)

        self.custom_aperture_mask = mask

    def create_grating_mask(
        self, settings: GratingSettings, x_axis: bool = True, y_axis: bool = False,
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

    def apply_aperture_masks(self):
        from lens_simulation import utils

        # create if they dont exist
        if self.non_lens_mask is None:
            self.non_lens_mask = np.zeros_like(self.profile)
        if self.custom_aperture_mask is None:
            self.custom_aperture_mask = np.zeros_like(self.profile)
        if self.truncation_aperture_mask is None:
            self.truncation_aperture_mask = np.zeros_like(self.profile)
        if self.sim_aperture_mask is None:
            self.sim_aperture_mask = np.zeros_like(self.profile)

        # match shape
        if self.non_lens_mask.shape != self.profile.shape:
            self.non_lens_mask = utils.pad_to_equal_size(
                self.non_lens_mask, self.profile
            ).astype(bool)

        if self.custom_aperture_mask.shape != self.profile.shape:
            self.custom_aperture_mask = utils.pad_to_equal_size(
                self.custom_aperture_mask, self.profile
            ).astype(bool)

        if self.truncation_aperture_mask.shape != self.profile.shape:
            self.truncation_aperture_mask = utils.pad_to_equal_size(
                self.truncation_aperture_mask, self.profile
            ).astype(bool)

        if self.sim_aperture_mask.shape != self.profile.shape:
            self.sim_aperture_mask = utils.pad_to_equal_size(
                self.sim_aperture_mask, self.profile
            ).astype(bool)

        # combine aperture masks
        self.aperture = (
            self.non_lens_mask
            + self.truncation_aperture_mask
            + self.custom_aperture_mask
            + self.sim_aperture_mask
        ).astype(bool)

    def create_escape_path(self, parameters, ep: float) -> None:
        """Create the escape path for the lens

        Args:
            lens (Lens): lens
            parameters (SimulationParameters): simulation parameters
            ep (float): escape path percentage

        Returns:
            np.ndarray: lens profile with escape path
        """
        from lens_simulation import utils

        if self.lens_type not in [LensType.Cylindrical, LensType.Spherical]:
            raise ValueError(
                f"Lens type {self.lens_type} is not supported for escape paths."
            )

        lens_h, lens_w = self.profile.shape

        # TODO: decide if we want independent escape path amounts.
        ep_h, ep_w = calculate_escape_path_dimensions(self, ep)

        # check if escape path fits within the simulation size
        test_escape_path_fits_inside_simulation(self, parameters, ep)

        ESCAPE_PATH_VALUE = 0

        # padding dimensions
        pad_h = int((ep_h - lens_h) // 2)
        pad_w = int((ep_w - lens_w) // 2)

        # use the maxium dimension to pad escape path?
        # TODO: decide if we use the maximum dimension for calculating the escape path or individual
        # TODO: if we use the maxium padding amount, we need to check it fits in both dimensions
        # pad_d = max(pad_h, pad_w)

        profile_with_escape_path = np.pad(
            self.profile,
            pad_width=((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=ESCAPE_PATH_VALUE,
        )
        
        if self.lens_type is LensType.Spherical:

            assert ep_h == ep_w, f"escape path must be symmetric {ep_h}, {ep_w}"
            assert lens_h == lens_w, f"lens must be symmetric {lens_h}, {lens_w}"

            distance = utils.create_distance_map_px(ep_w, ep_h)
            self.escape_mask = (distance <= ep_h // 2) * (distance >= lens_h // 2)
            profile_with_escape_path[self.escape_mask] = 0

            # set the non-lens_area
            self.non_lens_mask = (distance > ep_h // 2).astype(bool)
            profile_with_escape_path[self.non_lens_mask] = 0

        self.profile = profile_with_escape_path


def create_profile_1d(
    diameter: float, height: float, exponent: float, n_pixels: int
) -> np.ndarray:
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


def calculate_grating_coords(
    profile: np.ndarray, settings: GratingSettings, axis: int = 1,
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
    grating_centre_coords_x2 = np.arange(
        start_coord_2, profile.shape[axis], settings.distance_px
    )
    grating_centre_coords = sorted(
        np.append(grating_centre_coords_x1, grating_centre_coords_x2)
    )

    # coordinates of full grating which is grating centre +/- width / 2
    grating_coords = []

    for px in grating_centre_coords:

        min_px, max_px = px - settings.width_px / 2, px + settings.width_px / 2

        grating = np.arange(min_px, max_px).astype(int)

        grating = np.clip(grating, 0, profile.shape[axis] - 1)

        grating_coords.append(grating)

    return np.ravel(grating_coords)


def generate_lens(lens_config: dict, medium: Medium) -> Lens:
    """Generate a lens from a dictionary configuration

    Args:
        lens_config (dict): _description_
        medium (Medium): _description_

    Returns:
        Lens: _description_
    """

    lens_config = validation._validate_default_lens_config(lens_config)

    lens = Lens(diameter=lens_config["diameter"],
                height=lens_config["height"],
                exponent=lens_config["exponent"], 
                medium=medium, 
                lens_type=LensType[lens_config["lens_type"]])

    return lens
    


def apply_modifications(lens: Lens, lens_config: dict, parameters) -> Lens:
    """Apply all lens modifications defined in config"""

    if lens_config["escape_path"] is not None:
        lens.create_escape_path(parameters, lens_config["escape_path"])

    if lens_config["grating"] is not None:
        grating_settings = GratingSettings(
            width=lens_config["grating"]["width"],
            distance=lens_config["grating"]["distance"],
            depth=lens_config["grating"]["depth"],
            centred=lens_config["grating"]["centred"],
        )
        lens.create_grating_mask(
            grating_settings,
            x_axis=lens_config["grating"]["x"],
            y_axis=lens_config["grating"]["y"],
        )

    if lens_config["truncation"] is not None:
        lens.create_truncation_mask(
            truncation=lens_config["truncation"]["height"],
            radius=lens_config["truncation"]["radius"],
            type=lens_config["truncation"]["type"],
        )

    if lens_config["aperture"] is not None:
        lens.create_custom_aperture(
            inner_m=lens_config["aperture"]["inner"],
            outer_m=lens_config["aperture"]["outer"],
            type=lens_config["aperture"]["type"],
            inverted=lens_config["aperture"]["invert"],
        )

    # apply masks
    use_grating = True if lens_config["grating"] is not None else False
    use_truncation = True if lens_config["truncation"] is not None else False
    use_aperture = True if lens_config["aperture"] is not None else False

    lens.apply_masks(
        grating=use_grating, truncation=use_truncation, aperture=use_aperture,
    )

    return lens


def calculate_escape_path_dimensions(lens: Lens, ep: float):
    """Calculate the maxium dimensions for the escape path"""

    lens_h, lens_w = lens.profile.shape
    ep_h, ep_w = int(lens_h * (1 + ep)), int(lens_w * (1 + ep))

    return (ep_h, ep_w)


def test_escape_path_fits_inside_simulation(lens: Lens, parameters, ep: float):
    from lens_simulation import utils
    from lens_simulation.structures import SimulationParameters

    n_pixels_sim_height = utils._calculate_num_of_pixels(
        parameters.sim_height, parameters.pixel_size
    )
    n_pixels_sim_width = utils._calculate_num_of_pixels(
        parameters.sim_width, parameters.pixel_size
    )

    # calculate the escape path dimensions
    ep_h, ep_w = calculate_escape_path_dimensions(lens, ep)

    if ep_h > n_pixels_sim_height:
        raise ValueError(
            f"The given escape path is outside the simulation size: ep: {ep_h}px, sim: {n_pixels_sim_height}px"
        )

    if ep_w > n_pixels_sim_width:
        raise ValueError(
            f"The given escape path is outside the simulation size: ep: {ep_w}px, sim: {n_pixels_sim_width}px"
        )


def check_modification_masks(lens):
    from lens_simulation import utils

    # create if they dont exist
    if lens.grating_mask is None:
        lens.grating_mask = np.zeros_like(lens.profile)
    if lens.escape_mask is None:
        lens.escape_mask = np.zeros_like(lens.profile)
    if lens.truncation_mask is None:
        lens.truncation_mask = np.zeros_like(lens.profile)

    # match shape
    if lens.grating_mask.shape != lens.profile.shape:
        lens.grating_mask = utils.pad_to_equal_size(
            lens.grating_mask, lens.profile
        ).astype(bool)

    if lens.escape_mask.shape != lens.profile.shape:
        lens.escape_mask = utils.pad_to_equal_size(
            lens.escape_mask, lens.profile
        ).astype(bool)

    if lens.truncation_mask.shape != lens.profile.shape:
        lens.truncation_mask = utils.pad_to_equal_size(
            lens.truncation_mask, lens.profile
        ).astype(bool)

    return lens
