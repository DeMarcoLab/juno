METRE_TO_MM = 1e3
MM_TO_METRE = 1e-3

METRE_TO_MICRON = 1e6
MICRON_TO_METRE = 1e-6

METRE_TO_NANO = 1e9
NANO_TO_METRE = 1e-9

BEAM_SWEEPABLE_KEYS = ["width", "height", 
                "position_x", "position_y", 
                "theta", "numerical_aperture", 
                "tilt_x", "tilt_y", 
                "source_distance", "final_width", "focal_multiple"]


LENS_SWEEPABLE_KEYS = [
    "medium", "diameter", "height", "exponent"
]

MODIFICATION_SWEEPABLE_KEYS = [    
    "width", "distance", "depth", 
    "height", "radius",
    "inner", "outer"]

STAGE_SWEEPABLE_KEYS = [
    "output", "start_distance", "finish_distance", "focal_distance_start_multiple", "focal_distance_multiple" 
]
