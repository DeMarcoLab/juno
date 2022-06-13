from dataclasses import dataclass
import numpy as np

from scipy import ndimage
from enum import Enum


# TODO: 488 comes from sim
# TODO: fix the __repr__ for Medium
# TODO: comparison not working for dataclass?
@dataclass
class Medium:
    def __init__(self, refractive_index: float = 1.0, wavelength: float = 488.e-9) -> None:
        self.refractive_index = refractive_index
        self.wavelength: float = wavelength / refractive_index
        self.wave_number: float = 2 * np.pi / wavelength


@dataclass
class Water(Medium):
    refractive_index: float = 1.33


@dataclass
class Air(Medium):
    refactive_index: float = 1.00


@dataclass
class LithiumNiobate(Medium):
    refractive_index: float = 2.348


# TODO: add more common mediums.. (and names)