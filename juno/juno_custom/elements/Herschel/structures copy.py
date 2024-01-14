from dataclasses import dataclass

import numpy as np


@dataclass
class HerschelSurfaces:
    lens_points_first: np.ndarray = None
    lens_surface_first: np.ndarray = None
    lens_points_second: np.ndarray = None
    lens_surface_second: np.ndarray = None

    @staticmethod
    def from_dict(surfaces):
        return HerschelSurfaces(
            lens_points_first=surfaces["lens_points_first"],
            lens_surface_first=surfaces["lens_surface_first"],
            lens_points_second=surfaces["lens_points_second"],
            lens_surface_second=surfaces["lens_surface_second"],
        )

    @staticmethod
    def to_dict(surfaces):
        return {
            "lens_points_first": surfaces.lens_points_first,
            "lens_surface_first": surfaces.lens_surface_first,
            "lens_points_second": surfaces.lens_points_second,
            "lens_surface_second": surfaces.lens_surface_second,
        }


@dataclass
class HerschelLenses:
    lens_first: np.ndarray = None
    lens_second: np.ndarray = None
    lens_points_first: np.ndarray = None
    lens_points_second: np.ndarray = None
    lens_padding_first: np.ndarray = None
    lens_padding_second: np.ndarray = None


@dataclass
class HerschelSettings:
    pixel_size: float = 0.0001  # pixel size of the simulation
    lens_radius: float = 0.1  # radius of the chosen lens surface
    inner_radius: float = 0.01  # inner radius of the chosen lens surface
    lens_thickness: float = 0.01  # thickness of the lens
    lens_exponent: float = 2  # shape of the chosen lens surface
    z_medium_o: float = 1  # distance from the input plane to the lens
    z_medium_i: float = 1  # distance from the lens to the output plane
    n_medium_o: float = 1  # refractive index of the input medium
    n_medium_i: float = 1  # refractive index of the output medium
    n_lens: float = 2.348  # refractive index of the lens
    magnification: float = 2  # axial magnification of the lens

    a: float = 0  # initial guess for the root finding method
    tolerance: float = (
        1e-6  # tolerance value to evaluate success of root finding method
    )
    max_iter: int = 200000  # maximum number of iterations for the root finding method

    # input_lens_height: float = None # height of the input lens
    # input_lens_exponent: float = 2 # shape of the input lens
    # input_lens_medium: float = 1.5 # refractive index of the input lens medium
    # divergent: bool = False # whether the lens is divergent or not

    radii: np.ndarray = None  # array of lens radii to be used in the calculation

    @staticmethod
    def from_dict(settings):
        return HerschelSettings(
            pixel_size=settings["pixel_size"],
            lens_radius=settings["lens_radius"],
            inner_radius=settings["inner_radius"],
            lens_thickness=settings["lens_thickness"],
            lens_exponent=settings["lens_exponent"],
            z_medium_o=settings["z_medium_o"],
            z_medium_i=settings["z_medium_i"],
            n_medium_o=settings["n_medium_o"],
            n_medium_i=settings["n_medium_i"],
            n_lens=settings["n_lens"],
            magnification=settings["magnification"],
            a=settings["a"],
            tolerance=settings["tolerance"],
            max_iter=settings["max_iter"] if "max_iter" in settings else 200000,
            # input_lens_height=settings['input_lens_height'] if 'input_lens_height' in settings else 1,
            # input_lens_exponent=settings['input_lens_exponent'] if 'input_lens_exponent' in settings else 2,
            # input_lens_medium=settings['input_lens_medium'] if 'input_lens_medium' in settings else 1.5,
            # divergent=settings['divergent'] if 'divergent' in settings else False,
            radii=np.linspace(
                settings["inner_radius"],
                settings["lens_radius"],
                int(
                    (settings["lens_radius"] - settings["inner_radius"])
                    / settings["pixel_size"]
                )
                + 1,
            ),
        )

    @staticmethod
    def to_dict(settings):
        return {
            "pixel_size": settings.pixel_size,
            "lens_radius": settings.lens_radius,
            "inner_radius": settings.inner_radius,
            "lens_thickness": settings.lens_thickness,
            "lens_exponent": settings.lens_exponent,
            "z_medium_o": settings.z_medium_o,
            "z_medium_i": settings.z_medium_i,
            "n_medium_o": settings.n_medium_o,
            "n_medium_i": settings.n_medium_i,
            "n_lens": settings.n_lens,
            "magnification": settings.magnification,
            "a": settings.a,
            "tolerance": settings.tolerance,
            "max_iter": settings.max_iter,
            # 'input_lens_height': settings.input_lens_height,
            # 'input_lens_exponent': settings.input_lens_exponent,
            # 'input_lens_medium': settings.input_lens_medium,
            # 'divergent': settings.divergent,
            "radii": None,
        }


# @dataclass
# class HerschelParameters():
#     a: float = 0 # initial guess for the root finding method
#     tolerance: float = 1e-6 # tolerance value to evaluate success of root finding method
#     pixel_size: float = 0.0001 # size of the pixels in the simulation
#     start: float = 0.01 # start value (percentage) for the lens radius
#     stop: float = 1 # stop value (percentage) for the lens radius
#     step: float = 0.02 # step value (percentage) for the lens radius

#     max_iter: int = 200000 # maximum number of iterations for the root finding method
#     wavelength: float = 0.0005 # wavelength of the light
#     sim_width: float = None # width of the simulation area
#     lens_zero: dict = None # dictionary of the zeroth lens surface
#     lens_first: dict = None # dictionary of the first lens surface
#     lens_second: dict = None # dictionary of the second lens surface


#     @staticmethod
#     def from_dict(parameters):
#         return HerschelParameters(
#             a=parameters['a'],
#             tolerance=parameters['tolerance'],
#             pixel_size=parameters['pixel_size'],
#             start=parameters['start'],
#             stop=parameters['stop'],
#             step=parameters['step'],
#             radii=np.linspace(
#                 parameters['start'],
#                 parameters['stop'],
#                 int((parameters['stop'] - parameters['start']) / parameters['step']) + 1
#                 ),
#             max_iter=parameters['max_iter'] if 'max_iter' in parameters else 200000,
#             wavelength=parameters['wavelength'] if 'wavelength' in parameters else 0.0005,
#             sim_width=parameters['sim_width'] if 'sim_width' in parameters else None,
#             lens_zero=parameters['lens_zero'] if 'lens_zero' in parameters else None,
#             lens_first=parameters['lens_first'] if 'lens_first' in parameters else None,
#             lens_second=parameters['lens_second'] if 'lens_second' in parameters else None
#         )

#     @staticmethod
#     def to_dict(parameters):
#         return {
#             'a': parameters.a,
#             'tolerance': parameters.tolerance,
#             'pixel_size': parameters.pixel_size,
#             'start': parameters.start,
#             'stop': parameters.stop,
#             'step': parameters.step,
#             'radii': None,
#             'max_iter': parameters.max_iter,
#             'wavelength': parameters.wavelength,
#             'sim_width': parameters.sim_width,
#             'lens_zero': parameters.lens_zero,
#             'lens_first': parameters.lens_first,
#             'lens_second': parameters.lens_second
#         }


@dataclass
class HerschelResults:
    lens_roots: np.ndarray = None  # the roots of the lens surface
    lens_points_first: np.ndarray = None  # the points of the first lens surface
    lens_surface_first: np.ndarray = None  # the first lens surface
    lens_points_second: np.ndarray = None  # the points of the second lens surface
    lens_surface_second: np.ndarray = None  # the second lens surface
