


def _validate_lens_config(lens_config: dict, medium_dict: dict) -> dict:
    """Validate the lens configuration"""
    
    if lens_config["medium"] not in medium_dict:
        raise ValueError("Lens Medium not found in simulation mediums")
    
    # default settings
    lens_config["length"] = None if "length" not in lens_config else lens_config["length"]

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