import matplotlib.pyplot as plt
import numpy as np
from lens_simulation import Lens, Simulation, utils
from scipy import fftpack
from matplotlib import pyplot as plt
from enum import Enum, auto

amplitude = 10000
sim_wavelength = 488e-9
sim_width = 150e-6
# pixel_size = 0.1e-6
pixel_size = 0.02e-6
output_medium = Lens.Medium(1.)

# Beam settings chosen by user
beam_width = 130e-6
beam_height = 2.269158e-6
beam_position = [-0e-6, 0e-6]

beam = Lens.Lens(
    diameter=beam_width,
    height=beam_height,
    exponent=1,
    medium=Lens.Medium(2.348)
)

beam.generate_profile(pixel_size=pixel_size, lens_type=Lens.LensType.Spherical)

# set up the part of the lens square that isn't the lens for aperturing
non_lens_profile = beam.profile == 0
aperturing_value = -1e-9
beam.profile[non_lens_profile] = aperturing_value

# calculate padding parameters
pad_width = (int(sim_width/pixel_size)-len(beam.profile))//2 + 1
relative_position_x = int(beam_position[0]/pixel_size)
relative_position_y = int(beam_position[1]/pixel_size)

# pad the profile to the sim width (Top - Bottom - Left - Right)
beam.profile = np.pad(beam.profile, ((pad_width + relative_position_y, pad_width - relative_position_y),
                                             (pad_width + relative_position_x, pad_width - relative_position_x)))

# set up the sim padding to be apertured
beam.profile[:, :(pad_width + relative_position_y)] = aperturing_value
beam.profile[:, -(pad_width - relative_position_y):] = aperturing_value
beam.profile[:(pad_width + relative_position_x), :] = aperturing_value
beam.profile[-(pad_width - relative_position_x):, :] = aperturing_value

start_distance = 0
finish_distance = 7e-3

# regular delta calculation
delta = (beam.medium.refractive_index-output_medium.refractive_index) * beam.profile
# regular phase calculation
phase = (2 * np.pi * delta / sim_wavelength) #% (2 * np.pi)

# regular wavefront calculation
wavefront = amplitude * np.exp(1j * phase)

# asymmetric aperturing (apply aperture mask)
wavefront[beam.profile==aperturing_value] = 0 + 0j

# regular wavefront FFT
wavefront = fftpack.fft2(wavefront)

# regular frequency array creation
frequency_array = Simulation.generate_sq_freq_arr(sim_profile=beam.profile, pixel_size=pixel_size)

n_slices_1 = 20

sim = np.ones(shape=((n_slices_1), len(beam.profile[0]), len(beam.profile[1])))
distances_1 = np.linspace(start_distance, finish_distance, n_slices_1)
for i, z in enumerate(distances_1):
    prop_1 = np.exp(1j * output_medium.wave_number * z) * np.exp(
        (-1j * 2 * np.pi ** 2 * z * frequency_array) / output_medium.wave_number
    )
    # print("prop shape: ", prop.shape)
    propagation = fftpack.ifft2(prop_1 * wavefront)

    output = np.sqrt(propagation.real ** 2 + propagation.imag ** 2)

    sim[i] = np.round(output, 10)

sim_to_show = sim[:, sim.shape[1]//2, :]

utils.plot_simulation(sim_to_show, sim_to_show.shape[1], sim_to_show.shape[0], pixel_size, start_distance, finish_distance)

sim_to_show = sim[:, :, sim.shape[-1]//2]

utils.plot_simulation(sim_to_show, sim_to_show.shape[1], sim_to_show.shape[0], pixel_size, start_distance, finish_distance)

plt.show()
