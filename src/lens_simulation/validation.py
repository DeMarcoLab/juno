


def _validate_lens_config(lens_config: dict, medium_dict: dict) -> dict:
    """Validate the lens configuration"""
    
    if lens_config["medium"] not in medium_dict:
        raise ValueError("Lens Medium not found in simulation mediums")
    
    lens_config = _validate_default_lens_config(lens_config)

    return lens_config

def _validate_default_lens_config(lens_config: dict) -> dict:
    
    # required settings
    if "name" not in lens_config:
        raise ValueError(f"Lens config requires name. None provided")

    if "diameter" not in lens_config:
        raise ValueError(f"Lens config requires diameter. None provided")

    if "height" not in lens_config:
        raise ValueError(f"Lens config requires height. None provided")
    
    if "exponent" not in lens_config:
        raise ValueError(f"Lens config requires exponent. None provided")

    # default settings
    lens_config["length"] = None if "length" not in lens_config else lens_config["length"]
    lens_config["custom"] = None if "custom" not in lens_config else lens_config["custom"]
    lens_config["grating"] = None if "grating" not in lens_config else lens_config["grating"]
    lens_config["truncation"] = None if "truncation" not in lens_config else lens_config["truncation"]
    lens_config["aperture"] = None if "aperture" not in lens_config else lens_config["aperture"]

    return lens_config

def _validate_simulation_stage_config(stages: list, medium_dict: dict, lens_dict: dict) -> None:
    """Validate that all lenses and mediums have been defined, and all simulation stages have been
    defined correctly.
    """

    for stage in stages:
        # validate all lens, mediums exist, 
        if  stage["output"] not in medium_dict:
            raise ValueError(f"{stage['output']} has not been defined in the configuration")
        if stage["lens"] not in lens_dict:
            raise ValueError(f"{stage['lens']} has not been defined in the configuration")

        stage = _validate_simulation_stage(stage)


    return stages

def _validate_simulation_stage(stage: dict) -> dict:
   
    # validate simulation settings
    if "n_slices" not in stage and "step_size" not in stage:
        raise ValueError(f"Stage config requires n_slices or step_size")
    if "start_distance" not in stage:
        raise ValueError(f"Stage config requires start_distance")
    if "finish_distance" not in stage:
        raise ValueError(f"Stage config requires finish_distance")

    # default settings
    stage["n_slices"] = None if "n_slices" not in stage else stage["n_slices"]
    stage["step_size"] = None if "step_size" not in stage else stage["step_size"]

    return stage


def _validate_sim_config(config: dict):

    if "sim_parameters" not in config:
        raise ValueError(f"Simulation config requires sim_parameters. Not provided.")

    if "options" not in config:
        raise ValueError(f"Simulation config requires options. Not provided.")

    if "beam" not in config:
        raise ValueError(f"Simulation config requires beam. Not provided.")

    if "lenses" not in config:
        raise ValueError(f"Simulation config requires lenses. Not provided.")

    if "mediums" not in config:
        raise ValueError(f"Simulation config requires mediums. Not provided.")

    if "stages" not in config:
        raise ValueError(f"Simulation config requires stages. Not provided.")

    return config