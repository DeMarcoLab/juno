from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage


# TODO: comparison not working for dataclass?
@dataclass
class Medium:
    def __init__(self, refractive_index: float = 1.0, wavelength: float = 488.e-9) -> None:
        self.refractive_index = refractive_index
        self.wavelength: float = wavelength / refractive_index # equivalent wavelength
        self.wave_number: float = 2 * np.pi / self.wavelength


@dataclass
class Water(Medium):
    refractive_index: float = 1.33


@dataclass
class Air(Medium):
    refactive_index: float = 1.00


@dataclass
class LithiumNiobate(Medium):
    refractive_index: float = 2.348
