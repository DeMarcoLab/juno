from dataclasses import dataclass

import numpy as np
from juno.Lens import Lens


@dataclass
class HerschelSettings:
    """Class to hold the settings for generating the Herschel element."""

    radius: float = 9  # radius of the first lens surface
    radius_buffer: float = 0.01  # starting point for calculations of first lens surface
    thickness: float = 15  # nominal thickness of the lens
    exponent: float = 2  # initial guess of the exponent of the lens
    z_medium_o: float = (
        -30
    )  # position of the design point source relative to the first lens surface
    z_medium_i: float = 60  # position of the design final image point relative to the second lens surface
    n_medium_o: float = (
        1  # refractive index of the medium before the first lens surface
    )
    n_medium_i: float = (
        1  # refractive index of the medium after the second lens surface
    )
    n_lens: float = 2.0  # refractive index of the lens
    magnification: float = 2  # axial magnification of the lens
    a: float = 0  # initial guess for the root finding method
    tolerance: float = 1  # tolerance value to evaluate success of root finding method
    max_iter: int = 2000  # maximum number of iterations for the root finding method

    radii: np.ndarray = None  # array of radii of first lens surface

    @staticmethod
    def from_dict(settings):
        return HerschelSettings(
            radius=settings["radius"],
            radius_buffer=settings["radius_buffer"],
            thickness=settings["thickness"],
            exponent=settings["exponent"],
            z_medium_o=settings["z_medium_o"],
            z_medium_i=settings["z_medium_i"],
            n_medium_o=settings["n_medium_o"],
            n_medium_i=settings["n_medium_i"],
            n_lens=settings["n_lens"],
            magnification=settings["magnification"],
            a=settings["a"],
            tolerance=settings["tolerance"],
            max_iter=settings["max_iter"] if "max_iter" in settings else 200000,
        )

    def to_dict(settings):
        return {
            "radius": settings.radius,
            "radius_buffer": settings.radius_buffer,
            "thickness": settings.thickness,
            "exponent": settings.exponent,
            "z_medium_o": settings.z_medium_o,
            "z_medium_i": settings.z_medium_i,
            "n_medium_o": settings.n_medium_o,
            "n_medium_i": settings.n_medium_i,
            "n_lens": settings.n_lens,
            "magnification": settings.magnification,
            "a": settings.a,
            "tolerance": settings.tolerance,
            "max_iter": settings.max_iter,
        }

    def calculate_radii(self, pixel_size):
        self.radii = np.linspace(
            self.radius_buffer,
            self.radius,
            int((self.radius - self.radius_buffer) / pixel_size) + 1,
        )


@dataclass
class HerschelProfilesRaw:
    """Class to hold the intermediate results of generating the Herschel element."""

    roots: np.ndarray = None  # the roots of the lens surface
    x_first: np.ndarray = None  # the radial points of the first lens surface
    y_first: np.ndarray = None  # the height values of the first lens surface
    x_second: np.ndarray = None  # the radial points of the second lens surface
    y_second: np.ndarray = None  # the height values of the second lens surface


@dataclass
class HerschelProfiles:
    """Class to hold the final results of generating the Herschel element."""

    x_first: np.ndarray = None  # Juno lens object for the first lens surface
    y_first: np.ndarray = None  # Juno lens object for the second lens surface
    x_second: np.ndarray = None  # points of the first lens surface
    y_second: np.ndarray = None  # points of the second lens surface


@dataclass
class HerschelLenses:
    """Class to hold Herschel Juno lens objects"""

    first: Lens = None  # Juno lens object for the first lens surface
    second: Lens = None  # Juno lens object for the second lens surface


@dataclass
class HerschelLensesPadded:
    """Class to hold Herschel Juno lens objects"""

    first: Lens = None  # Juno lens object for the first lens surface
    second: Lens = None  # Juno lens object for the second lens surface
    first_padding: np.ndarray = None  # padding for the first lens surface
    second_padding: np.ndarray = None  # padding for the second lens surface


# Used for custom simulating, would require a lot of refactoring to get base Juno compatible with this so taking some shortcuts temporarily
@dataclass
class HerschelSimSettings:
    initial_lens_radius: float = 0.5  # initial radius of the lens
    initial_lens_focal_length: float = 0.5  # initial focal length of the lens
    initial_lens_exponent: float = 2  # initial exponent of the lens
    n_medium_o_sim: float = 1  # refractive index of the input medium
    n_medium_i_sim: float = 1  # refractive index of the output medium
    n_lens_sim: float = 1.5  # refractive index of the lens
    wavelength: float = 0.0005  # wavelength of the light
    sim_width: float = None  # width of the simulation area

    lens_zero: dict = None  # dictionary of the zeroth lens surface
    lens_first: dict = None  # dictionary of the first lens surface
    lens_second: dict = None  # dictionary of the second lens surface

    @staticmethod
    def from_dict(settings):
        return HerschelSimSettings(
            initial_lens_radius=settings["initial_lens_radius"]
            if "initial_lens_radius" in settings
            else 0.5,
            initial_lens_focal_length=settings["initial_lens_focal_length"]
            if "initial_lens_focal_length" in settings
            else 0.5,
            initial_lens_exponent=settings["initial_lens_exponent"]
            if "initial_lens_exponent" in settings
            else 2,
            n_medium_o_sim=settings["n_medium_o_sim"]
            if "n_medium_o_sim" in settings
            else 1,
            n_medium_i_sim=settings["n_medium_i_sim"]
            if "n_medium_i_sim" in settings
            else 1,
            n_lens_sim=settings["n_lens_sim"] if "n_lens_sim" in settings else 1.5,
            wavelength=settings["wavelength"] if "wavelength" in settings else 0.0005,
            sim_width=settings["sim_width"] if "sim_width" in settings else None,
            lens_zero=settings["lens_zero"] if "lens_zero" in settings else None,
            lens_first=settings["lens_first"] if "lens_first" in settings else None,
            lens_second=settings["lens_second"] if "lens_second" in settings else None,
        )

    @staticmethod
    def to_dict(settings):
        return {
            "initial_lens_radius": settings.initial_lens_radius,
            "initial_lens_focal_length": settings.initial_lens_focal_length,
            "initial_lens_exponent": settings.initial_lens_exponent,
            "n_medium_o_sim": settings.n_medium_o_sim,
            "n_medium_i_sim": settings.n_medium_i_sim,
            "n_lens_sim": settings.n_lens_sim,
            "wavelength": settings.wavelength,
            "sim_width": settings.sim_width,
            "lens_zero": settings.lens_zero,
            "lens_first": settings.lens_first,
            "lens_second": settings.lens_second,
        }
