# conversions
METRE_TO_MM = 1e3
MM_TO_METRE = 1e-3

METRE_TO_MICRON = 1e6
MICRON_TO_METRE = 1e-6

METRE_TO_NANO = 1e9
NANO_TO_METRE = 1e-9

# sweepable keys
BEAM_SWEEPABLE_KEYS = [
    "width",
    "height",
    "position_x",
    "position_y",
    "theta",
    "numerical_aperture",
    "tilt_x",
    "tilt_y",
    "source_distance",
    "final_diameter",
    "focal_multiple",
    "gaussian_wx", "gaussian_wy", "gaussian_z0", "gaussian_z"
]
LENS_SWEEPABLE_KEYS = ["medium", "diameter", "height", "exponent"]
CUSTOM_PROFILE_KEY = "custom"
CUSTOM_CONFIG_KEY = "custom_config"
GRATING_SWEEPABLE_KEYS = [
    "width",
    "distance",
    "depth",
]
TRUNCATION_SWEEPABLE_KEYS = ["height", "radius"]
APERTURE_SWEEPABLE_KEYS = ["inner", "outer"]
MODIFICATION_SWEEPABLE_KEYS = [*GRATING_SWEEPABLE_KEYS, *TRUNCATION_SWEEPABLE_KEYS, *APERTURE_SWEEPABLE_KEYS]
STAGE_SWEEPABLE_KEYS = [
    "output",
    "start_distance",
    "finish_distance",
    "focal_distance_start_multiple",
    "focal_distance_multiple",
]

# required config keys
REQUIRED_SIMULATION_KEYS = ["sim_parameters", "options", "beam", "lenses", "stages"]
REQUIRED_SIMULATION_OPTIONS_KEYS = ["log_dir"]
REQUIRED_SIMULATION_PARAMETER_KEYS = [
    "A",
    "pixel_size",
    "sim_height",
    "sim_width",
    "sim_wavelength",
]
REQUIRED_SIMULATION_STAGE_KEYS = ["lens", "output"]
REQUIRED_BEAM_KEYS = ["width", "height"]
REQUIRED_LENS_KEYS = ["name", "medium", "diameter", "height", "exponent"]
REQUIRED_LENS_GRATING_KEYS = ["width", "distance", "depth", "x", "y", "centred","mode", "blur", "inner_radius"]
REQUIRED_LENS_TRUNCATION_KEYS = ["height", "radius", "type", "aperture"]
REQUIRED_LENS_APERTURE_KEYS = ["inner", "outer", "type", "invert"]
