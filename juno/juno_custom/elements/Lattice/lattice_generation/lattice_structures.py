import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from juno.Simulation import generate_sq_freq_arr, propagate_over_distance
from scipy import fftpack

from juno_custom.lattice_generation import lattice_utils


@dataclass
class LatticeParameters:
    wavelength: float
    pixel_size: float
    wavelength_medium: float
    n_o: float
    n_l: float
    realspace_x: float
    realspace_y: float

    @staticmethod
    def from_dict(lattice_parameters: dict):
        return LatticeParameters(
            wavelength=lattice_parameters["wavelength"],
            pixel_size=lattice_parameters["pixel_size"],
            wavelength_medium=lattice_parameters["wavelength"]/lattice_parameters["n_o"],
            n_o=lattice_parameters["n_o"],
            n_l=lattice_parameters["n_l"],
            realspace_x=lattice_parameters["realspace_x"],
            realspace_y=lattice_parameters["realspace_y"]
        )
    
@dataclass
class LatticeData:
    phase: np.ndarray = None
    wavefront: np.ndarray = None
    pupil: np.ndarray = None
    tranpose: bool = False
    parameters: LatticeParameters = None

    @staticmethod
    def from_dict(lattice_data: dict):
        phase_ = lattice_data.get("phase")
        wavefront_ = lattice_data.get("wavefront")
        pupil_ = lattice_data.get("pupil")

        if wavefront_ is None and phase_ is None and pupil_ is None:
            raise ValueError("No wavefront or phase or pupil data provided")
        
        if wavefront_:
            if wavefront_.endswith(".pkl"):
                with open(lattice_data["wavefront"], "rb") as file:
                    wavefront = pickle.load(file)

        elif phase_ and phase_.endswith(".pkl"):
            with open(lattice_data["phase"], "rb") as file:
                phase = pickle.load(file)
            wavefront = lattice_utils.wavefront_from_phase(phase)

        elif pupil_ and pupil_.endswith(".pkl"):
            with open(lattice_data["pupil"], "rb") as file:
                pupil = pickle.load(file)
            wavefront = lattice_utils.wavefront_from_pupil(pupil)
    
        return LatticeData(
            phase=None,
            tranpose=lattice_data.get("transpose", False),
            wavefront=wavefront,
            pupil=None,
            parameters=LatticeParameters.from_dict(lattice_data["parameters"])
        )


class Lattice():
    def __init__(self, lattice_data: LatticeData) -> None:
        self.data = lattice_data
        self.delta_map = None
        self.intensity = None
        self.height_profile = None
        self.rounded_output = None
        self.propagation = None
        if self.data.tranpose:
            print("Transposing wavefront")
            self.data.wavefront = np.transpose(self.data.wavefront)
        self.update_wavefront(self.data.wavefront)
        self.calculate_profile()

    def update_wavefront(self, wavefront):
        self.data.wavefront = wavefront
        self.pupil_from_wavefront()
        self.data.wavefront = fftpack.ifftshift(fftpack.ifft2((self.data.pupil)))
        self.data.wavefront /= np.max(np.abs(self.data.wavefront))
        self.calculate_intensity()
        self.calculate_profile()


    def pupil_from_wavefront(self):
        self.data.pupil = np.abs(fftpack.fft2(self.data.wavefront))
        diff = 20
        self.data.pupil = fftpack.fftshift(self.data.pupil)
        cy, cx = self.data.pupil.shape[0] // 2, self.data.pupil.shape[1] // 2
        self.data.pupil[cy-diff:cy+diff, cx-diff:cx+diff] = 0
        # self.data.pupil[0, 0] = 0
        # self.data.pupil = fftpack.fftshift(self.data.pupil)

    def calculate_phase_map(self):
        self.data.phase = np.angle(self.data.wavefront)

    def calculate_delta_map(self):
        self.delta_map = self.data.phase / (2 * np.pi / self.data.parameters.wavelength)

    def calculate_intensity(self):
        self.intensity = np.abs(self.data.wavefront)**2

    def calculate_height_profile(self):
        self.height_profile = self.delta_map / (self.data.parameters.n_o - self.data.parameters.n_l)

    def calculate_profile(self):
        self.calculate_phase_map()
        self.calculate_delta_map()
        self.calculate_height_profile()

    def plot(self, cmap=None):
        self.plot_grid(self.data.phase, self.intensity, np.log(self.data.pupil+1e-5), names=["Phase", "Intensity", "Pupil"], cmap=cmap)
    
    @staticmethod
    def plot_grid(*args, **kwargs):
        names = kwargs.get("names")
        cmap = kwargs.get("cmap", "gray")
        fig, axes = plt.subplots(1, len(args), figsize=(50/len(args), 10/len(args)))
        for i, arg in enumerate(args):
            axes[i].imshow(arg, cmap=cmap)
            axes[i].set_title("{}".format(names[i]))
            axes[i].set_xlabel("x/λ")
            axes[i].set_ylabel("z/λ")
            axes[i].set_aspect("equal")
            fig.colorbar(axes[i].imshow(arg, cmap=cmap), ax=axes[i])
        plt.show()

    def calculate_propagation(self, distance):
        fft_wavefront = fftpack.fftshift(fftpack.fft2(self.data.wavefront))
        freq_arr = generate_sq_freq_arr(self.data.wavefront, self.data.parameters.pixel_size)
        wavenumber = 2 * np.pi / self.data.parameters.wavelength
        self.rounded_output, self.propagation = propagate_over_distance(
            fft_wavefront, distance, freq_arr, wavenumber
        )

    def plot_propagation(self, distance):
        self.calculate_propagation(distance)
        plt.imshow(self.data.rounded_output, 
                   cmap="gray", aspect="auto", 
                   extent=[-self.data.parameters.realspace_x/2, 
                           self.data.parameters.realspace_x/2, 
                           -self.data.parameters.realspace_y/2, 
                           self.data.parameters.realspace_y/2])
        plt.title("Stationary LLS xz PSF {}lambda".format(distance/self.data.parameters.wavelength))
        plt.xlabel("x/λ")
        plt.ylabel("z/λ")
        plt.colorbar()
        plt.show()


