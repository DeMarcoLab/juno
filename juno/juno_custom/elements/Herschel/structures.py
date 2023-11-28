from dataclasses import dataclass

import numpy as np
from juno.Lens import Lens

@dataclass
class HerschelSettings:
    """Class to hold the settings for generating the Herschel element."""
    radius: float = 9 # radius of the first lens surface
    radius_buffer: float = 0.01 # starting point for calculations of first lens surface
    thickness: float = 15 # nominal thickness of the lens
    exponent: float = 2 # initial guess of the exponent of the lens
    z_medium_o: float = -30 # position of the design point source relative to the first lens surface
    z_medium_i: float = 60 # position of the design final image point relative to the second lens surface
    n_medium_o: float = 1 # refractive index of the medium before the first lens surface
    n_medium_i: float = 1 # refractive index of the medium after the second lens surface
    n_lens: float = 2.  # refractive index of the lens
    magnification: float = 2  # axial magnification of the lens
    a: float = 0 # initial guess for the root finding method
    tolerance: float = 1 # tolerance value to evaluate success of root finding method
    max_iter: int = 2000 # maximum number of iterations for the root finding method
    
    radii: np.ndarray = None # array of radii of first lens surface


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
            int((self.radius - self.radius_buffer)/ pixel_size) + 1,
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
    x_first: np.ndarray = None # Juno lens object for the first lens surface
    y_first: np.ndarray = None # Juno lens object for the second lens surface
    x_second: np.ndarray = None # points of the first lens surface
    y_second: np.ndarray = None # points of the second lens surface

@dataclass
class HerschelLenses:
    """Class to hold Herschel Juno lens objects"""
    first: Lens = None # Juno lens object for the first lens surface
    second: Lens = None # Juno lens object for the second lens surface