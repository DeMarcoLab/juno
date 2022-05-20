import matplotlib.pyplot as plt
import numpy as np
from lens_simulation import Lens, Simulation, utils
from scipy import fftpack
from matplotlib import pyplot as plt
from enum import Enum, auto

from lens_simulation.Beam import (
    BeamSpread,
    DistanceMode,
    BeamShape,
    theta_from_NA,
    focal_distance_from_theta,
    height_from_focal_distance,
)

'================================================================================='

def main():
    amplitude = 10000
    sim_wavelength = 488e-9
    sim_width = 1500e-6
    pixel_size = 10e-6
    output_medium = Lens.Medium(1.33)

    # Beam settings chosen by user
    beam_width = 15e-6
    beam_height = 5e-6
    # beam_width = 150e-6
    # beam_height = 50e-6
    beam_width = 500e-6
    beam_height = 150e-6
    beam_position = [-300e-6, 200e-6]
    beam_position = [-0e-6, 0e-6]

    n_slices_1 = 20

    # this is selected by DistanceMode flag in config
    distance_mode = DistanceMode.Direct

    # type of beam spread
    beam_spread = BeamSpread.Plane

    # type of beam shape
    beam_shape = BeamShape.Square

    if beam_shape is not BeamShape.Rectangular:
        beam_height = beam_width

    if beam_spread is BeamSpread.Plane:
        distance_mode = DistanceMode.Direct

    # can't do converging/diverging square case
    if beam_spread is not BeamSpread.Plane:
        beam_shape = BeamShape.Circular

        # Theta can either be specified directly, or by numerical aperture (config flag probably)
        theta = np.deg2rad(10)
        # OR
        # theta = theta_from_NA(numerical_aperture=0.4, output_medium=output_medium)

    # perpendicular distance of source from the first lens can be defined directly (aperture distance)
    # or based on a desired beam width at the lens plane
    source_aperture_distance = 20000e-6
    final_beam_width = 800e-6
    if beam_spread is BeamSpread.Plane:
        final_beam_width = beam_width

    # Default beam specifications
    beam = Lens.Lens(
        diameter=max(beam_width, beam_height),
        height=100,
        exponent=2,
        medium=Lens.Medium(100)
    )

    beam.generate_profile(pixel_size=pixel_size, lens_type=Lens.LensType.Spherical)

    # generate the lens profile
    if beam_spread is BeamSpread.Plane:
        if beam_shape is BeamShape.Circular:
            beam.profile = (beam.profile != 0) * 1
        elif beam_shape is BeamShape.Square:
            beam.profile = np.ones(shape=beam.profile.shape)
        elif beam_shape is BeamShape.Rectangular:
            beam.profile = np.zeros(shape=beam.profile.shape)
            # make sure to fill out in the correct order, otherwise this creates a square
            if beam_height > beam_width:
                profile_width = int(beam_width/pixel_size/2)
                beam.profile[beam.profile.shape[0]//2-profile_width:beam.profile.shape[0]//2+profile_width, :] = 1
            elif beam_width > beam_height:
                profile_height = int(beam_height/pixel_size/2)
                beam.profile[:, beam.profile.shape[0]//2-profile_height:beam.profile.shape[0]//2+profile_height] = 1
    # diverging/converging cases
    else:
        # calculate the equivalent focal distance of the required convergence angle
        focal_distance = focal_distance_from_theta(beam=beam, theta=theta)
        # calculate and set the height of the apertures 'virtual' lens, re-generate the profile with new height
        beam.height = height_from_focal_distance(beam, output_medium=output_medium, focal_distance=focal_distance)
        print(focal_distance)
        print(beam.height)
        beam.generate_profile(pixel_size=pixel_size, lens_type=Lens.LensType.Spherical)

        if beam_spread is BeamSpread.Diverging:
            beam.invert_profile()
            beam.profile = (beam.profile < beam.height) * beam.profile

    # set up the part of the lens square that isn't the lens for aperturing
    non_lens_profile = beam.profile == 0
    aperturing_value = -1e-9
    beam.profile[non_lens_profile] = aperturing_value

    # plt.imshow(beam.profile)
    # plt.colorbar()

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

    # If you just want the source to be a certain distance away:
    if distance_mode is DistanceMode.Direct or beam_spread is BeamSpread.Plane:
        finish_distance = source_aperture_distance

    elif distance_mode is DistanceMode.Focal:
        finish_distance = focal_distance

    # if you want the beam to converge/diverge to a specific width
    elif distance_mode is DistanceMode.Width:
        final_beam_radius = final_beam_width/2
        if beam_spread is BeamSpread.Converging:
            finish_distance = focal_distance - (final_beam_radius/np.tan(theta))
        else:
            finish_distance = ((final_beam_radius-(beam.diameter/2))/np.tan(theta))
    else:
        raise ValueError('DistanceMode set incorrectly')

    print(finish_distance)

    tilt_enabled = True
    xtilt = 25 #-45
    ytilt = 75

    # regular delta calculation
    delta = (beam.medium.refractive_index-output_medium.refractive_index) * beam.profile

    # tilt the beam
    if tilt_enabled:
        x = np.arange(len(beam.profile))*pixel_size
        y = np.arange(len(beam.profile))*pixel_size

        x_tilt = x * np.tan(np.deg2rad(xtilt))
        y_tilt = y * np.tan(np.deg2rad(ytilt))


        # modify the optical path of the light based on tilt
        # delta = delta + np.outer(x, y)
        # delta = delta + x * np.tan(np.deg2rad(xtilt))
        delta = delta + np.add.outer(y * np.tan(np.deg2rad(ytilt)), -x * np.tan(np.deg2rad(xtilt)))

    # regular phase calculation
    phase = (2 * np.pi * delta / sim_wavelength) #% (2 * np.pi)

    # plt.imshow(np.outer(x, y))

    # plt.imshow(np.log(delta+1e-9))
    # plt.imshow(np.log(delta))
    # plt.imshow(phase)
    # plt.colorbar()

    # print(x_tilt)
    # print(y_tilt)
    # print(np.add.outer(x_tilt, y_tilt))
    # plt.imshow(np.add.outer(y_tilt, x_tilt))

    # regular wavefront calculation
    wavefront = amplitude * np.exp(1j * phase)

    # asymmetric aperturing (apply aperture mask)
    wavefront[beam.profile==aperturing_value] = 0 + 0j

    # regular wavefront FFT
    wavefront = fftpack.fft2(wavefront)

    # regular frequency array creation
    frequency_array = Simulation.generate_sq_freq_arr(sim_profile=beam.profile, pixel_size=pixel_size)

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

    sim_to_show = sim[-1]
    utils.plot_simulation(sim_to_show, sim_to_show.shape[1], sim_to_show.shape[0], pixel_size, start_distance, finish_distance)

    # for sim_to_show in sim[::2]:
    #     plt.figure()
    #     utils.plot_simulation(sim_to_show, sim_to_show.shape[1], sim_to_show.shape[0], pixel_size, start_distance, finish_distance)

    plt.show()

if __name__ == "__main__":
    main()