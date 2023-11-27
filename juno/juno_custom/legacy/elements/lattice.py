import numpy as np
from juno import utils as j_utils
from juno.Lens import Lens
from scipy import ndimage

from juno_custom.element_template import ElementTemplate
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifftshift, fftshift
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import interpolation
import os
import sys
from PIL import Image

from numba import jit

# name: [type, default, critical, fixed, description]
LATTICE_KEYS = {
    "wave": [float, 0.488, True, False, "Wavelength of light in um"],
    "NA_inner": [float, 0.42, True, False, "Inner NA of the mask aperture"],
    "NA_outer": [float, 0.6, True, False, "Outer NA of the mask aperture"],
    "spacing": [float, 0.97, True, False, "Spacing between the beams"],
    "n_beam": [int, 46, True, False, "Number of beams"],
    "crop": [float, 0.22, True, True, "Crop factor"],
    "tilt": [float, 0, True, False, "Tilt of the pattern"],
    "shift_x": [float, 0, True, False, "Shift of the pattern in x direction"],
    "shift_y": [float, 0, True, False, "Shift of the pattern in y direction"],
    "mag": [float, 167.364, True, False, "Magnification between SLM mask and sample"],
    "pixel": [float, 13.62, True, False, "Pixel size of the element in um"],
    "slm_xpix": [int, 1280, True, False, "Number of pixels in x direction"],
    "slm_ypix": [int, 1024, True, False, "Number of pixels in y direction"],
    "mode": [str, "binary", True, False, "Mode of the pattern (binary, pupil, intensity)"],
}


class Lattice(ElementTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Lattice"

    def __repr__(self) -> str:
        return f"""Lattice Profile"""

    def generate_profile(self, params):
        self.profile = linear_bessel_array(
            show=False,
            outdir=None,
            pattern_only=True,
            test=False,
            save=False,
            savefig=False,
            path=None,
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
    crop=0.22,
    tilt=0,
    shift_x=0,
    shift_y=0,
    mag=167.364,
    pixel=13.662,
    slm_xpix=1280,
    slm_ypix=1024,
    fillchip=0.95,
    fudge=0.95,
    show=False,
    outdir=None,
    pattern_only=True,
    test=False,
    save=False,
    savefig=False,
    mode="binary",
    path=None,
):

    if path:
        if save or savefig:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    # auto-choose good spacing
    if not spacing:
        spacing = fudge * wave / NA_inner

    # to fill the chip
    if n_beam == "fill" and fillchip:
        n_beam = int(
            np.floor(1 + ((fillchip * (slm_xpix * (pixel / mag) / 2)) / spacing))
        )

    # Populate real space array
    dx = pixel / mag
    x = np.arange(-(slm_xpix) / 2, (slm_xpix + 1) / 2, 1.0) * dx
    y = x

    # for scipy interpolation functions, we don't use the meshgrid...
    x_slm = np.linspace(x[0], x[-1], slm_xpix)
    y_slm = x_slm

    # Populate k space array
    dk = 2 * np.pi / (slm_xpix + 1) / dx
    kx = np.arange(-(slm_xpix) / 2, (slm_xpix + 1) / 2, 1.0) * dk
    ky = kx
    [kx, ky] = np.meshgrid(kx, ky)
    kr = np.sqrt(kx * kx + ky * ky)

    # Mask k-space array according to inner and outer NA
    pupil_mask = (kr < NA_outer * (2 * np.pi / wave)) & (
        kr > NA_inner * (2 * np.pi / wave)
    )

    if path:
        if save:
            np.save(os.path.join(path, "pupil_mask"), pupil_mask)
        if savefig:
            plt.figure()
            plt.imshow(pupil_mask)
            plt.title("Pupil mask")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "pupil_mask.png"))

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

    if path:
        if save:
            np.save(os.path.join(path, "pupil_field_ideal"), pupil_field_ideal)
        if savefig:
            plt.figure()
            plt.imshow(np.abs(pupil_field_ideal))
            plt.title("Pupil field ideal")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "pupil_field_ideal.png"))

    # Ideal SLM field of fourier transform of pupil field
    slm_field_ideal = fftshift(fft2(ifftshift(pupil_field_ideal))).real
    slm_field_ideal /= np.max(np.max(np.abs(slm_field_ideal)))

    # TODO: Save ideal slm field
    ideal = np.abs(slm_field_ideal * slm_field_ideal)

    if path:
        if save:
            np.save(os.path.join(path, "ideal"), ideal)
        if savefig:
            plt.figure()
            plt.imshow(ideal)
            plt.title("Ideal coherent bessel light sheet intensity")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "ideal.png"))

    # Display ideal intensity at sample (incorporates supersampling)
    if show:
        plt.figure()
        plt.imshow(np.abs(slm_field_ideal * slm_field_ideal))
        plt.title("Ideal coherent bessel light sheet intensity")
        plt.axis("image")
        plt.colorbar()

    # Interpolate back onto SLM pixels and apply cropping factor
    # interpolator = interp2d(x, x, slm_field_ideal)
    interpolator = RectBivariateSpline(x, y, slm_field_ideal)
    slm_pattern = interpolator(x_slm, y_slm)

    if path:
        if save:
            np.save(os.path.join(path, "slm_pattern"), slm_pattern)
        if savefig:
            plt.figure()
            plt.imshow(slm_pattern)
            plt.title("SLM pattern")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "slm_pattern.png"))

    # slm_pattern *= np.abs(slm_pattern) > crop
    slm_pattern[np.abs(slm_pattern) < crop] = 0
    eps = np.finfo(float).eps
    slm_pattern = np.sign(slm_pattern + eps) * np.pi / 2 + np.pi / 2

    if path:
        if save:
            np.save(os.path.join(path, "slm_pattern_binary"), slm_pattern)
        if savefig:
            plt.figure()
            plt.imshow(slm_pattern)
            plt.title("SLM pattern binarised")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "slm_pattern_binary.png"))

    # Account for rectangular aspect ratio of SLM and convert phase to binary
    low = int(np.floor((slm_xpix / 2) - (slm_ypix / 2) - 1))
    high = int(low + slm_ypix)
    slm_pattern_final = (slm_pattern[low:high, :] / np.pi) != 0

    if path:
        if save:
            np.save(os.path.join(path, "slm_pattern_binary_2"), slm_pattern_final)
        if savefig:
            plt.figure()
            plt.imshow(slm_pattern_final)
            plt.title("SLM pattern binary 2")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "slm_pattern_binary_2.png"))

    if outdir is not None:
        outdir = os.path.abspath(os.path.expanduser(outdir))
        if os.path.isdir(outdir):
            namefmt = (
                "{:.0f}_{:2d}b_s{:.2f}_c{:.2f}_na{:.0f}-{:.0f}_x{:02f}_y{:02f}_t{:0.3f}"
            )
            name = namefmt.format(
                wave * 1000,
                n_beam * 2 - 1,
                spacing,
                crop,
                100 * NA_outer,
                100 * NA_inner,
                shift_x,
                shift_y,
                tilt,
            )
            name = name.replace(".", "p")
            outpath = os.path.join(outdir, name + ".png")

            imout = Image.fromarray(slm_pattern_final.astype(np.uint8) * 255)
            imout = imout.convert("1")
            imout.save(outpath)

    if show:
        plt.figure()
        plt.imshow(slm_pattern, interpolation="nearest")
        plt.title(
            "Cropped and pixelated phase from SLM pattern exiting the polarizing beam splitter"
        )
        plt.axis("image")

        plt.figure()
        plt.imshow(slm_pattern_final, interpolation="nearest", cmap="gray")
        plt.title("Binarized image to output to SLM")

    # if pattern_only:
    #     print(f'Mode: {mode}')
    #     if mode == 'binary':
    #         return slm_pattern_final #, intensity_final, pupil_field
    #     elif mode == 'pupil':
    #         return pupil_field
    #     elif mode == 'intensity':
    #         return intensity_final
    #     if show:
    #         plt.show()
    #     return slm_pattern_final

    # this method uses nearest neighbor like the matlab version
    [xmesh, ymesh] = np.meshgrid(x, y)
    coords = np.array([xmesh.flatten(), ymesh.flatten()]).T
    interpolator = RegularGridInterpolator(
        (x_slm, y_slm), slm_pattern, method="nearest"
    )
    slm_pattern_cal = interpolator(coords)  # supposed to be nearest neighbor
    slm_pattern_cal = slm_pattern_cal.reshape(len(x), len(y)).T
    slm_field = np.exp(1j * slm_pattern_cal)

    # at this point, matlab has complex component = 0.0i

    # Compute intensity impinging on annular mask
    pupil_field_impinging = fftshift(fft2(ifftshift(slm_field)))
    pupil_field_impinging_real = np.real(
        pupil_field_impinging * np.conj(pupil_field_impinging)
    )
    if path:
        if save:
            np.save(
                os.path.join(path, "pupil_field_impinging_real"),
                pupil_field_impinging_real,
            )
        if savefig:
            plt.figure()
            plt.imshow(pupil_field_impinging_real)
            plt.title("Real component of impinging pupil field")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "pupil_field_impinging_real.png"))

    # Compute intensity passing through annular mask
    pupil_field = pupil_field_impinging * pupil_mask
    pupil_field_real = np.real(pupil_field * np.conj(pupil_field))
    if path:
        if save:
            np.save(os.path.join(path, "pupil_field_real"), pupil_field_real)
        if savefig:
            plt.figure()
            plt.imshow(pupil_field_real)
            plt.title("Real part of pupil field")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "pupil_field_real.png"))

    if show:
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(
            (pupil_field_impinging * np.conj(pupil_field_impinging)).real,
            interpolation="nearest",
            cmap="inferno",
        )
        plt.clim(0, (2 * n_beam - 1) * 3e6)
        plt.title("Intensity impinging on annular mask")
        plt.subplot(1, 2, 2, sharex=ax1)
        plt.imshow(
            (pupil_field * np.conj(pupil_field)).real,
            interpolation="nearest",
            cmap="inferno",
        )
        plt.clim(0, (2 * n_beam - 1) * 3e6)
        plt.title("Intensity after annular mask")

    # Compute intensity at sample
    field_final = fftshift(fft2(ifftshift(pupil_field)))
    intensity_final = (field_final * np.conj(field_final)).real

    if path:
        if save:
            np.save(os.path.join(path, "sample_intensity"), intensity_final)
        if savefig:
            plt.figure()
            plt.imshow(intensity_final)
            plt.title("Sample Intensity")
            plt.axis("image")
            plt.colorbar()
            plt.savefig(os.path.join(path, "sample_intensity.png"))

    if show:
        plt.figure()
        plt.imshow(intensity_final, interpolation="nearest")
        plt.title("Actual intensity at sample")
        plt.axis("image")
        plt.show()

    plt.close("all")
    if test:
        return pupil_field, slm_pattern_final, intensity_final

    # print(f"Mode: {mode}")
    if mode == "binary":
        return slm_pattern_final  # , intensity_final, pupil_field
    elif mode == "pupil":
        return pupil_field_real
    elif mode == "intensity":
        return intensity_final
