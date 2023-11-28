import numpy as np

from juno.juno_custom.elements.element_template import ElementTemplate
from juno.juno_custom.elements.Herschel import utils as h_utils
from juno.juno_custom.elements.Herschel.structures import HerschelSettings

# name: [type, default, critical, fixed, description]
HERSCHEL_KEYS = {
    "radius": [float, 9, True, False, "Radius of the first element "],
    "radius_buffer": [float, 0.05, True, False, "starting point for calculations of first lens surface"],
    "thickness": [float, 15., True, False, "nominal thickness of the lens"],
    "exponent": [float, 2, True, False, "initial guess of the exponent of the lens"],
    "z_medium_o": [float, -30., True, False, "position of the design point source relative to the first lens surface"],
    "z_medium_i": [float, 60., True, False, "position of the design final image point relative to the second lens surface"],
    "n_medium_o": [float, 1, True, False, "refractive index of the medium before the first lens surface"],
    "n_medium_i": [float, 1, True, False, "refractive index of the medium after the second lens surface"],
    "n_lens": [float, 2, True, False, "refractive index of the lens"],
    "magnification": [float,  2 , True, False, "axial magnification of the lens"],
    "a": [float, 0, True, False, "initial guess for the root finding method"],
    "tolerance": [float,  0.1, True, False, "tolerance value to evaluate success of root finding method"],
    "max_iter": [int,  2000, True, False, "maximum number of iterations for the root finding method"],
}


class Herschel(ElementTemplate):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Herschel"

    def __repr__(self) -> str:
        return f"""Herschel"""

    def generate_profile(self, params):
        self.profile = self.generate_lens(**params)
        # self.generate_display_profile()

    def generate_lens(
        self,
        radius,
        radius_buffer,
        thickness,
        exponent,
        z_medium_o,
        z_medium_i,
        n_medium_o,
        n_medium_i,
        n_lens,
        magnification,
        a,
        tolerance,
        max_iter,

    ):
        self.settings = HerschelSettings.from_dict(
            {
                "radius": radius,
                "radius_buffer": radius_buffer,
                "thickness": thickness,
                "exponent": exponent,
                "z_medium_o": z_medium_o,
                "z_medium_i": z_medium_i,
                "n_medium_o": n_medium_o,
                "n_medium_i": n_medium_i,
                "n_lens": n_lens,
                "magnification": magnification,
                "a": a,
                "tolerance": tolerance,
                "max_iter": max_iter,
            }
        )

        raw_profiles = h_utils.create_raw_profiles(settings=self.settings)
        profiles = h_utils.calculate_profiles(settings=self.settings, raw_profiles=raw_profiles)
        lenses = h_utils.generate_lenses(settings=self.settings, profiles=profiles)

        return {
            "lens_first": lenses.first,
            "lens_second": lenses.second,
        }

    def generate_display_profile(self):
        import copy

        # Expect two surfaces, lens_first and lens_second, separated by lens_thickness
        # get the absolute difference between two numbers, a and b.  Accounting for the sign of each number
        first_profile = copy.deepcopy(self.profile["lens_first"])
        second_profile = copy.deepcopy(self.profile["lens_second"])

        height_1 = np.max(self.profile["lens_first"]) - np.min(
            self.profile["lens_first"]
        )
        height_2 = np.max(self.profile["lens_second"]) - np.min(
            self.profile["lens_second"]
        )
        thickness = self.settings.lens_thickness * self.settings.scale.value
        total_height = height_1 + height_2 + thickness
        max_dim = np.max(
            [self.profile["lens_first"].shape[1], self.profile["lens_second"].shape[1]]
        )
        # conditionally pad the smaller surface to match
        if self.profile["lens_first"].shape[1] < max_dim:
            pad_width = (max_dim - self.profile["lens_first"].shape[1]) // 2
            first_profile = np.pad(
                first_profile,
                ((0, 0), (pad_width, pad_width)),
                mode="constant",
                constant_values=np.max(self.profile["lens_first"]),
            )

        if self.profile["lens_second"].shape[1] < max_dim:
            pad_width = (max_dim - self.profile["lens_second"].shape[1]) // 2
            second_profile["lens_second"] = np.pad(
                second_profile,
                ((0, 0), (pad_width, pad_width)),
                mode="constant",
                constant_values=0,
            )

        # TODO: remove magic numbers
        print(self.profile["lens_first"].max())
        print(self.profile["lens_first"].min())
        print(f"height_1: {height_1}")
        print(f"height_2: {height_2}")
        print(f"thickness: {thickness}")
        print(f"total_height: {total_height}")

        z_step_size = 0.1 * self.settings.scale.value
        z_steps = int(total_height / z_step_size)
        print(f"z_step_size: {z_step_size}")
        print(f"z_steps: {z_steps}")

        profile_1_3D = utils.profile_to_3D(
            first_profile, z_steps=height_1 / z_step_size
        )
        profile_1_3D = 1 - profile_1_3D
        profile_2_3D = utils.profile_to_3D(
            second_profile, z_steps=height_2 / z_step_size
        )

        # # z_steps, y_steps, x_steps
        thickness_3D = (
            np.ones(shape=(int(thickness / z_step_size / 1), 1, int(max_dim))) * 0.0
        )

        # stack the three profiles along axis 0
        self.display_profile = np.concatenate(
            (profile_1_3D, thickness_3D, profile_2_3D), axis=0
        )
        print(f"display_profile.shape: {self.display_profile.shape}")

    @staticmethod
    def __keys__() -> list:
        return HERSCHEL_KEYS
