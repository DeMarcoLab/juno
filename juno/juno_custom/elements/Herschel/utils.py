import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from juno import utils
from juno.juno_custom.elements.Herschel import equations as eq
from juno.juno_custom.elements.Herschel.structures import *
from juno.Lens import Lens, LensSettings, LensType
from juno.Medium import Medium


def create_raw_profiles(settings: HerschelSettings) -> HerschelProfilesRaw:
    a_list = []
    z_a_list = []
    r_b_list = []
    z_b_list = []

    for point in settings.radii:
        # print(f"Calculating for radius {point}...")
        settings.radius = point
        eq_19_ = eq.eq_19(settings)
        root = optimize.newton(
            eq_19_, 0, tol=settings.tolerance, maxiter=settings.max_iter
        )

        a_list.append(root)

        z_a_ = eq.z_a(settings)
        z_a_list.append(z_a_(root))

        r_b_ = eq.r_b(settings)
        r_b_list.append(r_b_(root))

        z_b_ = eq.z_b(settings)
        z_b_list.append(z_b_(root))

    return HerschelProfilesRaw(
        roots=a_list,
        x_first=settings.radii,
        y_first=z_a_list,
        x_second=r_b_list,
        y_second=z_b_list,
    )


def calculate_profiles(
    settings: HerschelSettings, raw_profiles: HerschelProfilesRaw, pixel_size: float
) -> HerschelProfiles:
    x_first = np.concatenate(
        (-np.flip(raw_profiles.x_first), [0], raw_profiles.x_first)
    )
    y_first = np.concatenate((np.flip(raw_profiles.y_first), [0], raw_profiles.y_first))
    y_first = np.expand_dims(y_first, axis=0)

    y_second = np.concatenate(
        (np.flip(raw_profiles.y_second), [settings.thickness], raw_profiles.y_second)
    )

    x_second = np.concatenate(
        (-np.flip(raw_profiles.x_second), [0], raw_profiles.x_second)
    )

    x_second = np.round(x_second / pixel_size) * pixel_size

    lens_second_diameter = x_second[-1] - x_second[0]
    n_pixels = utils._calculate_num_of_pixels(lens_second_diameter, pixel_size)

    # re-interpolate the second lens to have the same pixel size as the first lens
    pixels = np.linspace(x_second[0], x_second[-1], n_pixels)

    y_second = np.interp(pixels, x_second, y_second)
    x_second = pixels
    y_second = np.expand_dims(y_second, axis=0)

    y_second = y_second - y_second.min()
    y_first = y_first - y_first.min()

    return HerschelProfiles(
        x_first=x_first,
        y_first=y_first,
        x_second=pixels,
        y_second=y_second,
    )


def generate_lenses(
    settings: HerschelSettings,
    profiles: HerschelProfiles,
) -> HerschelLenses:
    x_second = profiles.x_second

    y_first = profiles.y_first
    y_second = profiles.y_second

    y_first = y_first - y_first.min()
    y_second = y_second - y_second.min()

    first_lens_settings = LensSettings(
        diameter=settings.radius * 2,
        height=y_first.max(),
        exponent=2,
        medium=Medium(settings.n_medium_o),
        lens_type=LensType.Cylindrical,
    )
    second_lens_settings = LensSettings(
        diameter=settings.radius * 2,
        height=y_first.max(),
        exponent=2,
        medium=Medium(settings.n_lens),
        lens_type=LensType.Cylindrical,
    )

    first = Lens(
        diameter=settings.radius * 2,
        height=y_first.max(),
        exponent=2,
        medium=Medium(settings.n_medium_o),
        lens_type=LensType.Cylindrical,
        settings=first_lens_settings,
    )
    second = Lens(
        diameter=x_second.max() * 2,
        height=y_second.max(),
        exponent=2,
        medium=Medium(settings.n_lens),
        lens_type=LensType.Cylindrical,
        settings=second_lens_settings,
    )

    first.profile = y_first
    second.profile = y_second

    return HerschelLenses(
        first=first,
        second=second,
    )


def calculate_padding(
    profiles: HerschelProfiles,
    lenses: HerschelLensesPadded,
    settings: HerschelSettings,
    sim_settings: HerschelSimSettings,
    pixel_size: float,
):
    if sim_settings.sim_width is not None:
        n_pixels = utils._calculate_num_of_pixels(sim_settings.sim_width, pixel_size)
        # n_pixels = parameters.sim_width//parameters.step
        pad_width_first = (n_pixels - lenses.first.profile.shape[1]) // 2
        pad_width_second = (n_pixels - lenses.second.profile.shape[1]) // 2
        pad_width_second_points = (n_pixels - profiles.x_second.shape[0]) // 2
        lens_first = np.pad(
            lenses.first.profile,
            ((0, 0), (pad_width_first, pad_width_first)),
            mode="constant",
            constant_values=0,
        )
        lens_second = np.pad(
            lenses.second.profile,
            ((0, 0), (pad_width_second, pad_width_second)),
            mode="constant",
            constant_values=0,
        )
        lens_points_first = np.pad(
            profiles.x_first,
            (pad_width_first, pad_width_first),
            mode="constant",
            constant_values=np.NaN,
        )
        lens_points_second = np.pad(
            profiles.x_second,
            (pad_width_second_points, pad_width_second_points),
            mode="constant",
            constant_values=np.NaN,
        )

        # create arrays that are the same length as lens_points_first and lens_points_second
        lens_padding_first = np.zeros(lens_first.shape)
        lens_padding_second = np.zeros(lens_second.shape)

        # where these arrays were padded, set to 1 in the padding arrays
        lens_padding_first[0][:pad_width_first] = 1
        lens_padding_first[0][-pad_width_first:] = 1
        lens_padding_second[0][:pad_width_second] = 1
        lens_padding_second[0][-pad_width_second:] = 1

        lenses.first.profile = lens_first
        lenses.second.profile = lens_second

        return HerschelLensesPadded(
            first=lenses.first,
            second=lenses.second,
            first_padding=lens_padding_first,
            second_padding=lens_padding_second,
            # first_padding=lens_points_first,
            # second_padding=lens_points_second,
        )


def display_ray_tracing(
    settings: HerschelSettings, results: HerschelProfilesRaw, buffer: float = 1.1
):
    x_max = np.amax([results.x_first[-1], results.x_second[-1]]) * (buffer + 0.1)

    plt.figure()
    plt.xlim([-0.1 * x_max, x_max])
    plt.ylim(
        [
            settings.z_medium_o * buffer,
            (settings.z_medium_i + settings.thickness) * buffer,
        ]
    )
    skip = 20

    for i, point in enumerate(results.y_first[::skip]):
        i = i * skip
        o_x = [0, results.x_first[i]]
        o_y = [settings.z_medium_o, point]
        i_x = [0, results.x_second[i]]
        i_y = [
            settings.z_medium_i + settings.thickness,
            results.y_second[i],
        ]
        x = [results.x_first[i], results.x_second[i]]
        y = [point, results.y_second[i]]
        plt.plot(x, y, linestyle="--")
        plt.plot(o_x, o_y, linestyle="dotted")
        plt.plot(i_x, i_y, linestyle="dotted")

    plt.plot(results.x_first, results.y_first, linewidth=2, color="red")
    plt.plot(
        results.x_second,
        results.y_second,
        linewidth=2,
        color="red",
    )

    plt.figure()
    for i, point in enumerate(results.y_first[::skip]):
        i = i * skip
        o_x = [0, results.x_first[i]]
        o_y = [settings.z_medium_o, 0]
        i_x = [0, results.x_second[i]]
        i_y = [
            settings.z_medium_i + settings.thickness,
            settings.thickness,
        ]
        x = [results.x_first[i], results.x_second[i]]
        y = [0, settings.thickness]
        plt.plot(x, y, linestyle="--")
        plt.plot(o_x, o_y, linestyle="dotted")
        plt.plot(i_x, i_y, linestyle="dotted")

    plt.show()
