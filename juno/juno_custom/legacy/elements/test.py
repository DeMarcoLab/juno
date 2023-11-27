import numpy as np
from juno import utils as j_utils
from juno.Lens import Lens
from scipy import ndimage

from juno_custom.element_template import ElementTemplate

TEST_KEYS = {
    "pixel_size": [float, 1.e-6, True, False, "Pixel Size"],
    "diameter": [float, 0.42, True, False, "Diameter of the element"],
    "coefficient": [float, 0.5, True, False, "Coefficient of the element"],
    "escape_path": [float, 0.97, True, False, "Percentage of escape path"],
    "exponent": [int, 46, True, False, "Exponent of the element"],
}

class TestLens(ElementTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Test"

    def __repr__(self) -> str:
        return f"""Test"""

    def generate_profile(self, params):
        self.profile = generate_element(**params)

    def analyse(self):
        print("Test analysis")

    @staticmethod
    def __keys__() -> list:
        return TEST_KEYS


def generate_element(pixel_size, diameter, coefficient, escape_path, exponent):
    profile = np.zeros((1000, 1000))
    profile[450:550, 450:550] = 1
    return profile