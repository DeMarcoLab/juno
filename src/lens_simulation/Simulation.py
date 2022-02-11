
import numpy as np

from scipy import fftpack
import pylab


class Simulation:
    pass



def generate_frequency_array(n_pixels: int, pixel_size: float) -> np.ndarray:

    f_u = fftpack.fftfreq(n_pixels, 1)

    frequency_array = f_u ** 2

    return frequency_array


def generate_frequency_array_custom(n_pixels: int, pixel_size: float) -> np.ndarray:
    """n_pixels: number of pixels in the profile"""
    space_size = pixel_size * n_pixels 
    du = 1 / space_size
    U = pylab.arange(-n_pixels / 2, n_pixels / 2  , 1) * du

    U = U.astype(np.float32)
    f_u = fftpack.fftshift(U)

    frequency_array = f_u ** 2

    return frequency_array



