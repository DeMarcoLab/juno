import itertools
from math import ceil
from pathlib import Path
import uuid
import os
import petname

from pprint import pprint
import numpy as np
from tqdm import tqdm

from lens_simulation import Simulation, utils

# TODO: convert print to logging, and save log file
# TODO: add datetime to sim metadata?
# TODO: add time taken to logging
# TODO: change n_slices to a common sim parameter

class SimulationRunner:

    def __init__(self, config_filename: str) -> None:
        
        self.run_id = uuid.uuid4()
        self.petname = petname.generate(3)

        self.config = utils.load_config(config_filename)

        # create logging directory
        log_dir = os.getcwd() # TODO: make user selectable
        self.data_path: Path = os.path.join(log_dir , "log",  str(self.petname))
        os.makedirs(self.data_path, exist_ok=True)

        # update metadata
        self.config["run_id"] = self.run_id

    def initialise_simulation(self) -> None :

        print("-"*50)
        print(f"\nSimulation Run: {self.petname} ({self.run_id})")
        print(f"Data: {self.data_path}")
        print("-"*50)

        all_lens_params = []
        for lens in self.config["lenses"]:
            # pprint(lens)

            lens_params = []
            # sweepable parameters
            for key in ["height", "exponent"]:

                param_sweep = generate_parameter_sweep(lens[key])
                lens_params.append(param_sweep)
            
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
        # print(f"\n{n_lens_configs}} Lens Configurations. {n_stage_configs} Stage Configurations")
        print(f"Generating {n_lens_configs * n_stage_configs} Simulation Configurations. ")
        
        self.simulation_configurations = []

        # loop through all lens, then all stage combinations?
        for lens_combo in tqdm(self.all_parameters_combinations_lens):

            for stage_combo in tqdm(self.all_parameters_combinations_stages, leave=False):
                
                # create lens combinations
                lens_combination = []
                for i, lens_config in enumerate(self.config["lenses"]):
                    
                    lens_dict = lens_config
                    lens_dict["height"] = lens_combo[i][0]   
                    lens_dict["exponent"] = lens_combo[i][1]
                    
                    lens_combination.append(lens_dict)
                
                # create stage combination
                stage_combination = []
                for j, stage in enumerate(self.config["stages"]):
                    stage_dict = {
                        "lens": stage["lens"], # TODO: replace with combo
                        "output": stage["output"], # TODO: replace with combo
                        "start_distance": stage_combo[j][0],  
                        "finish_distance": stage_combo[j][1],
                        "n_slices": stage["n_slices"],
                        "options": stage["options"],
                    }
                    stage_combination.append(stage_dict)

                # generate simulation config
                sim_config = {
                    "run_id": self.run_id, 
                    "run_petname": self.petname, 
                    "log_dir": self.data_path, 
                    "sim_parameters": self.config["sim_parameters"],
                    "beam": self.config["beam"],
                    "options": self.config["options"],
                    "mediums": self.config["mediums"], 
                    "lenses": lens_combination,
                    "stages": stage_combination
                }

                self.simulation_configurations.append(sim_config)

       
        print(f"Generated {len(self.simulation_configurations)} simulation configurations.")

        # save sim configurations
        utils.save_metadata(self.config, self.data_path)

    def run_simulations(self):
        print(f"\nRunning {len(self.simulation_configurations)} Simulations")

        print("----------- Simulation Summary -----------")
        print(f"Pixel Size: {self.config['sim_parameters']['pixel_size']:.1e}m")
        print(f"Simulation Width: {self.config['sim_parameters']['sim_width']:.1e}m")
        print(f"No. Stages: {len(self.config['stages'])}")
        print("------------------------------------------")

        for sim_config in tqdm(self.simulation_configurations):

            sim = Simulation.Simulation(sim_config)
            sim.run_simulation()


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