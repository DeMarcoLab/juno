import numpy as np
from juno import utils as j_utils
from juno.Lens import Lens
from scipy import ndimage

from juno_custom.element_template import ElementTemplate

EXTENDED_KEYS = {
    "pixel_size": [float, 1.e-6, True, False, "Pixel size of the element in m"],
    "diameter": [float, 100.e-6, True, False, "Diameter of the element in m"],
    "coefficient": [float, 10000, True, False, "Coefficient of the element"],
    "escape_path": [float, 0.1, True, False, "Percentage of escape path"],
    "exponent": [float, 2.3, True, False, "Exponent of the element"],
}


class ExtendedMicrolens(ElementTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.name = "ExtendedMicrolens"

    def __repr__(self) -> str:
        return f"""Extended Microlens"""

    def generate_profile(self, params):
        self.profile = generate_lens(
            **params
        )

    def analyse(self):
        print("Extended microlens capability")

    @staticmethod
    def __keys__() -> list:
        return EXTENDED_KEYS


def generate_lens(pixel_size, diameter, coefficient, escape_path, exponent):
    n_pixels = j_utils._calculate_num_of_pixels(diameter, pixel_size)
    radius = diameter / 2
    n_pixels_in_radius = n_pixels // 2 + 1
    radius_px = np.linspace(0, radius, n_pixels_in_radius)
    profile = -coefficient * radius_px**exponent
    profile -= np.min(profile)
    profile = np.append(profile, np.zeros(int(escape_path * len(profile))))
    profile = np.append(np.flip(profile[1:]), profile)
    profile = ndimage.gaussian_filter(profile, sigma=3)
    profile = np.expand_dims(profile, 0).astype(np.float32)

    return profile
 