import numpy as np
from numba import jit
from numpy.fft import fft2, fftshift, ifftshift

from juno.juno_custom.elements.element_template import ElementTemplate

# name: [type, default, critical, fixed, description]
LATTICE_KEYS = {
    "wave": [float, 0.488, True, False, "Wavelength of light in um"],
    "NA_inner": [float, 0.42, True, False, "Inner NA of the mask aperture"],
    "NA_outer": [float, 0.6, True, False, "Outer NA of the mask aperture"],
    "spacing": [float, 0, True, False, "Spacing between the beams"],
    "n_beam": [int, 46, True, False, "Number of beams"],
    "tilt": [float, 0, True, False, "Tilt of the pattern"],
    "shift_x": [float, 0, True, False, "Shift of the pattern in x direction"],
    "shift_y": [float, 0, True, False, "Shift of the pattern in y direction"],
    "magnification": [float, 167.364, True, False, "Magnification between SLM mask and sample"],
    "pixel_size": [float, 13.62, True, False, "Pixel size of the element in um"],
    "n_pixels_x": [int, 1280, True, False, "Number of pixels in x direction"],
    "height": [float, 1.0, True, False, "Height of the element in um"],
}


class Lattice(ElementTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Lattice"

    def __repr__(self) -> str:
        return f"""Lattice Profile"""

    def generate_profile(self, params):
        self.profile = linear_bessel_array(
            **params,
        )

    @staticmethod
    def __keys__() -> dict:
        return LATTICE_KEYS


def linear_bessel_array(
    wave=0.488,
    NA_inner=0.44,
    NA_outer=0.55,
    spacing=None,
    n_beam="fill",
    tilt=0,
    shift_x=0,
    shift_y=0,
    magnification=167.364,
    pixel_size=13.662,
    n_pixels_x=1280,
    height=None,
):
    # auto-choose good spacing, defaulting to fudge of 0.95
    if not spacing or spacing == 0:
        spacing = 0.95 * wave / NA_inner

    # to fill the chip, defaulting to 0.95 fillchip value
    if n_beam == 0:
        n_beam = int(
            np.floor(
                1 + ((0.95 * (n_pixels_x * (pixel_size / magnification) / 2)) / spacing)
            )
        )

    # Populate real space array
    dx = pixel_size / magnification

    # Populate k space array
    dk = 2 * np.pi / (n_pixels_x + 1) / dx
    kx = np.arange(-(n_pixels_x) / 2, (n_pixels_x + 1) / 2, 1.0) * dk
    ky = kx
    [kx, ky] = np.meshgrid(kx, ky)
    kr = np.sqrt(kx * kx + ky * ky)

    # Mask k-space array according to inner and outer NA
    pupil_mask = (kr < NA_outer * (2 * np.pi / wave)) & (
        kr > NA_inner * (2 * np.pi / wave)
    )

    # Generate array of bessel beams by applying phase ramps in k-space
    pupil_field_ideal = pupil_mask.astype(np.complex128)

    f = kx * spacing * np.cos(tilt) + ky * spacing * np.sin(tilt)

    @jit(nopython=True)
    def calc(v, ii):
        A = np.exp(1j * f * ii) + np.exp(-1j * f * ii)
        return v + np.multiply(pupil_mask, A)

    for ii in range(1, n_beam):
        pupil_field_ideal = calc(pupil_field_ideal, ii)
    pupil_field_ideal *= np.exp(1j * (kx * shift_x + ky * shift_y))

    # Pupil field has been masked by this point

    # Ideal SLM field of fourier transform of pupil field
    slm_field_ideal = fftshift(fft2(ifftshift(pupil_field_ideal))).real
    slm_field_ideal /= np.max(np.max(np.abs(slm_field_ideal)))

    # SLM field is the 'ideal' filtered version
    # Don't think interpolation is required for actual phase as we don't use SLM

    slm_field = np.exp(1j * slm_field_ideal)

    # Compute intensity impinging on annular mask
    pupil_field_impinging = fftshift(fft2(ifftshift(slm_field)))
    # Compute intensity passing through annular mask
    pupil_field = pupil_field_impinging * pupil_mask
    pupil_field_real = np.real(pupil_field * np.conj(pupil_field))

    # Compute intensity at sample
    field_final = fftshift(fft2(ifftshift(pupil_field)))
    intensity_final = (field_final * np.conj(field_final)).real
    if height is not None:
        intensity_final /= intensity_final.max()
        intensity_final *= height

    return {
        "pupil_field_real": pupil_field_real,
        "intensity_final": intensity_final,
        "slm_field_ideal": slm_field_ideal,
        "field_final": field_final,
    }
