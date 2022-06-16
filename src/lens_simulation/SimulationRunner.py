import itertools
from math import ceil, floor
from pathlib import Path
import uuid
import os
import petname

import logging
from pprint import pprint
import numpy as np
from tqdm import tqdm

from lens_simulation import Simulation, utils
from lens_simulation import validation

from copy import deepcopy

# TODO: convert print to logging, and save log file

class SimulationRunner:

    def __init__(self, config_filename: str) -> None:
        
        self.run_id = uuid.uuid4()
        self.petname = petname.generate(3)

        self.config = utils.load_config(config_filename)

        # create logging directory
        self.data_path: Path = os.path.join(self.config["options"]["log_dir"], str(self.petname))
        os.makedirs(self.data_path, exist_ok=True)

        # update metadata
        self.config["run_id"] = self.run_id
        self.config["started"] = utils.current_timestamp()

        logfile = utils.configure_logging(self.data_path)

    def initialise_simulation(self) -> None :

        logging.info("-"*50)
        logging.info(f"Simulation Run: {self.petname} ({self.run_id})")
        logging.info(f"Data: {self.data_path}")
        logging.info("-"*50)

        all_lens_params = []
        for lens in self.config["lenses"]:
            # pprint(lens)

            lens_params = []
            # sweepable parameters
            for key in ["height", "exponent"]:

                param_sweep = generate_parameter_sweep(lens[key])
                lens_params.append(param_sweep)
            print(lens_params)
            # combinations for each lens
            # get all combinations of paramters
            # ref: https://stackoverflow.com/questions/798854/all-combinations-of-a-list-of-lists
            parameters_combinations = list(itertools.product(*lens_params))
            all_lens_params.append(parameters_combinations)
        
        # TODO: support sweeping through lens, and output for stages e.g. lens: [lens_2, lens_3, ...]
        all_stage_params = []
        for stage in self.config["stages"]:
            
            stage_params = [] 
            for key in ["start_distance", "finish_distance"]:
                
                # numeric sweep
                param_sweep = generate_parameter_sweep(stage[key])
                stage_params.append(param_sweep)
                # TODO: need to account for the use_focal_distance param, otherwise the distance sweeps are a waste...

            parameters_combinations = list(itertools.product(*stage_params))
            all_stage_params.append(parameters_combinations)

        # generate all combinations of all lens/stage parameters
        self.all_parameters_combinations_lens = list(itertools.product(*all_lens_params))
        self.all_parameters_combinations_stages = list(itertools.product(*all_stage_params))


    def setup_simulation(self):
        # TODO: this is currently hardcoded so only height and exponent can be swept

        # generate configuration for simulations
        n_lens_configs = len(self.all_parameters_combinations_lens)
        n_stage_configs = len(self.all_parameters_combinations_stages)
        # logging.info(f"{n_lens_configs}} Lens Configurations. {n_stage_configs} Stage Configurations")
        logging.info(f"Generating {n_lens_configs * n_stage_configs} Simulation Configurations. ")
        
        self.simulation_configurations = []

        # loop through all lens, then all stage combinations?
        for lens_combo in tqdm(self.all_parameters_combinations_lens):

            for stage_combo in tqdm(self.all_parameters_combinations_stages, leave=False):
                
                # create lens combinations
                lens_combination = []
                for i, lens_config in enumerate(self.config["lenses"]):
                    
                    simulation_lenses = lens_config
                    simulation_lenses["height"] = lens_combo[i][0]   
                    simulation_lenses["exponent"] = lens_combo[i][1]
                    
                    lens_combination.append(simulation_lenses)
                    del simulation_lenses
                
                # create stage combination
                stage_combination = []
                for j, stage in enumerate(self.config["stages"]):

                    stage = validation._validate_default_simulation_stage_config(stage)

                    stage_dict = {
                        "lens": stage["lens"], # TODO: replace with combo
                        "output": stage["output"], # TODO: replace with combo
                        "start_distance": stage_combo[j][0],  
                        "finish_distance": stage_combo[j][1],
                        "n_slices": stage["n_slices"],
                        "step_size": stage["step_size"],
                        "use_equivalent_focal_distance": stage["use_equivalent_focal_distance"],
                        "focal_distance_start_multiple": stage["focal_distance_start_multiple"],
                        "focal_distance_multiple": stage["focal_distance_multiple"]
                    }
                    stage_combination.append(stage_dict)
                    del stage_dict
                
                # generate simulation config
                sim_config = {
                    "run_id": self.run_id, 
                    "run_petname": self.petname, 
                    "log_dir": self.data_path, 
                    "sim_parameters": self.config["sim_parameters"],
                    "beam": self.config["beam"],
                    "options": self.config["options"], 
                    "lenses": deepcopy(lens_combination),
                    "stages": deepcopy(stage_combination)
                } 
                # NOTE: need to deepcopy due to way python dicts get updated...


                self.simulation_configurations.append(sim_config)

                del sim_config, lens_combination, stage_combination

        # BUG
        # after all the combinations are set correctly, they all get set to the last combination value
        # e.g. all lens exponents are set to the final exponent value...
        # must be something to do with how dictionaries are updated?

        # print("--------------------------- AFTER ---------------------------")
        # pprint([conf["lenses"] for conf in self.simulation_configurations[-3:]])

        logging.info(f"Generated {len(self.simulation_configurations)} simulation configurations.")

        # save sim configurations
        utils.save_metadata(self.config, self.data_path)

    def run_simulations(self):
        logging.info(f"Running {len(self.simulation_configurations)} Simulations")

        logging.info("----------- Simulation Summary -----------")
        logging.info(f"Pixel Size: {self.config['sim_parameters']['pixel_size']:.1e}m")
        logging.info(f"Simulation Wavelength: {self.config['sim_parameters']['sim_wavelength']}m")
        logging.info(f"Simulation Size: {self.config['sim_parameters']['sim_height']:.1e}m x {self.config['sim_parameters']['sim_width']:.1e}m")
        logging.info(f"No. Stages: {len(self.config['stages']) + 1}")
        logging.info("------------------------------------------")

        for sim_config in tqdm(self.simulation_configurations):

            sim = Simulation.Simulation(sim_config)
            sim.run_simulation()
            logging.info(f"Finished simulation {sim.petname}")
        
        # save final sim configuration
        logging.info(f"Finished running {len(self.simulation_configurations)} Simulations")
        self.config["finished"] = utils.current_timestamp()
        utils.save_metadata(self.config, self.data_path)



def generate_parameter_sweep_v2(start:float, stop:float, step_size:float) -> np.ndarray:

    if stop is None or step_size is None:
        return [start]

    if step_size == 0.0:
        raise ValueError("Step Size cannot be zero.")

    if start >= stop:
        raise ValueError(f"Start parameter {start} cannot be greater than stop parameter {stop}")

    SANTIY = +1e-12
    if (stop - start) + SANTIY < step_size:
        raise ValueError(f"Step size is larger than parameter range. {start}, {stop}, {step_size}")

    return np.arange(start, stop + SANTIY, step_size)



def generate_parameter_sweep(param: list) -> np.ndarray:
    # TODO: tests
    if isinstance(param, (float, int)):
        # single value parameter
        return [param]

    if isinstance(param, list):
        # single value list param
        if len(param) == 1:
            return param

    if len(param) != 3:
        # restrict parameter format
        raise RuntimeError("Parameters must be in the format [start, finish, step_size]")

    start, finish, step_size = param

    if step_size == 0.0:
        raise ValueError("Step Size cannot be zero.")

    if start >= finish:
        raise ValueError(f"Start parameter cannot be greater than finish parameter {param}")

    if (finish - start) + 0.0001e-6 < step_size:
        raise ValueError(f"Step size is larger than parameter range. {param}")

    n_steps = ceil((finish - start) / (step_size)) + 1 # TODO: validate this

    return np.linspace(start, finish, n_steps)


from lens_simulation.constants import LENS_SWEEPABLE_KEYS, MODIFICATION_SWEEPABLE_KEYS, BEAM_SWEEPABLE_KEYS, STAGE_SWEEPABLE_KEYS, GRATING_SWEEPABLE_KEYS, TRUNCATION_SWEEPABLE_KEYS, APERTURE_SWEEPABLE_KEYS

def sweep_config_keys(conf: dict, sweep_keys: list) -> list:
    key_params = []
    for k in sweep_keys:

        if isinstance(conf, dict) and k in conf: # check if param exists
            start, stop, step = conf[k], conf[f"{k}_stop"], conf[f"{k}_step"]
            params = generate_parameter_sweep_v2(start, stop, step)
        else:
            params = [None]
       
        key_params.append(params)
    return key_params


def generate_lens_parameter_combinations(config) -> list:

    all_lens_params = []
    for lc in config["lenses"]:

        lp = sweep_config_keys(lc, LENS_SWEEPABLE_KEYS)
        gp = sweep_config_keys(lc["grating"], GRATING_SWEEPABLE_KEYS)
        tp = sweep_config_keys(lc["truncation"], TRUNCATION_SWEEPABLE_KEYS)
        ap = sweep_config_keys(lc["aperture"], APERTURE_SWEEPABLE_KEYS)

        lens_param = [*lp, *gp, *tp, *ap]
        parameters_combinations = list(itertools.product(*lens_param))
        all_lens_params.append(parameters_combinations)
        
    all_parameters_combinations_lens = list(itertools.product(*all_lens_params))

    return all_parameters_combinations_lens


def get_lens_configurations(lpc: list, config: dict) -> list:
    lens_combination = []
    for i, lens_config in enumerate(config["lenses"]): # careful of the order here...

        simulation_lenses = lens_config
        simulation_lenses["medium"] = lpc[i][0]   
        simulation_lenses["diameter"] = lpc[i][1]
        simulation_lenses["height"] = lpc[i][2]   
        simulation_lenses["exponent"] = lpc[i][3]

        if simulation_lenses["grating"] is not None:
            simulation_lenses["grating"]["width"] = lpc[i][4]   
            simulation_lenses["grating"]["distance"] = lpc[i][5]
            simulation_lenses["grating"]["depth"] = lpc[i][6]   
        if simulation_lenses["truncation"] is not None:
            simulation_lenses["truncation"]["height"] = lpc[i][7]   
            simulation_lenses["truncation"]["radius"] = lpc[i][8]  
        if simulation_lenses["aperture"] is not None:         
            simulation_lenses["aperture"]["inner"] = lpc[i][9]   
            simulation_lenses["aperture"]["outer"] = lpc[i][10]   
    
        lens_combination.append(simulation_lenses)

    return lens_combination
