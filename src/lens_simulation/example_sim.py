from lens_simulation import Lens, Simulation

import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack


A = 10000
from lens_simulation import Simulation

sim_wavelength = 488e-9

# n_pixels need to be consistent

sim_width = 4500e-6
pixel_size = 1e-6  # 450e-6
# n_pixels = int(sim_width / pixel_size)

# # n_pixels must be odd (symmetry).
# if n_pixels % 2 == 0:
#     n_pixels += 1

# print(n_pixels)

lens = Lens.Lens(
    diameter=sim_width, height=70e-6, exponent=2.0, medium=Lens.Medium(2.348)
)
profile = lens.generate_profile(pixel_size)


# pixels size is defined by the sim
# define a simulation width (includes padding)
# TODO: minimum padding?
# escape path: 10%
# padding: 30%
# pad the sides of the profile to match the sim width

centre_px = (len(profile) - 1) // 2
print("CENTRE_PX: ", centre_px)

output_medium = Lens.Medium(refractive_index=1.5)

print("n_pixels_in_sim: ", len(profile), " pixel_size: ", pixel_size)

freq_array = Simulation.generate_squared_frequency_array(
    n_pixels=len(profile), pixel_size=pixel_size
)

delta = (lens.medium.refractive_index - output_medium.refractive_index) * profile
phase = (2 * np.pi * delta / sim_wavelength) % (2 * np.pi)
wavefront = A * np.exp(1j * phase)

# print("Wavefront shape: ", wavefront.shape)
wavefront = fftpack.fft(wavefront)

equivalent_focal_distance = Simulation.calculate_equivalent_focal_distance(lens, output_medium)
print(equivalent_focal_distance)
start_distance = 0. * equivalent_focal_distance  # 25e-3
finish_distance = 1. * equivalent_focal_distance  # 28e-3

n_slices = 1000
sim = np.ones(shape=(n_slices, len(profile)))
distances = np.linspace(start_distance, finish_distance, n_slices)
for i, z in enumerate(distances):

    prop = np.exp(1j * output_medium.wave_number * z) * np.exp(
        (-1j * 2 * np.pi ** 2 * z * freq_array) / output_medium.wave_number
    )
    # print("prop shape: ", prop.shape)
    propagation = fftpack.ifft(prop * wavefront)

    output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

    sim[i] = output

print("Prop Size: ", len(prop))

# print(lens)
# print(medium)

# fig, ax = plt.subplots(3, 1, figsize=(15, 15))

# ax[0].plot(profile)
# ax[0].set_title("PROFILE")

# ax[1].plot(phase)
# ax[1].set_title("PHASE")

# plt.show()

from lens_simulation import utils

# static simulation image
# fig = utils.plot_simulation(
#         sim,
#         width=200, height=100,
#         pixel_size_x=pixel_size,
#         start_distance=start_distance,
#         finish_distance=finish_distance)

# utils.save_figure(fig, "sim.png")

# plt.show()

# # interactive simulation
# fig = utils.plot_interactive_simulation(sim)
# fig.show()


# save simulation data

import numpy as np

# np.savez_compressed("sim.npz", sim.astype(np.float32)) #
np.save("sim.npy", sim)

del sim

sim = np.load("sim.npy")

import matplotlib.pyplot as plt


from lens_simulation import utils
utils.plot_simulation(sim, sim.shape[1], sim.shape[0], pixel_size, start_distance, finish_distance)
# plt.imshow(sim)
plt.show()

# fig = utils.plot_interactive_simulation(sim)
# fig.show()
