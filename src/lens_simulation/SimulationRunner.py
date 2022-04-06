from decimal import DivisionByZero
import itertools
from pathlib import Path
import uuid
import os
import petname

import datetime
import time
from pprint import pprint
import numpy as np
from tqdm import tqdm

from lens_simulation import Simulation, utils

# TODO: convert print to logging, and save log file
# TODO: allow parameter sweep for stage values? (lens, medium, distances...)

class SimulationRunner:

    def __init__(self, config_filename: str) -> None:
        self.data_path: Path
        self.run_id = uuid.uuid4()
        self.parameters = None
        self.petname = petname.generate(3)

        self.config = utils.load_config(config_filename)

        # create logging directory
        log_dir = os.getcwd() # TODO: make user selectable
        self.data_path = os.path.join(log_dir , "log",  str(self.petname))
        os.makedirs(self.data_path, exist_ok=True)

        # update metadata
        self.config["run_id"] = self.run_id

    def initialise_simulation(self) -> None :

        print("-"*50)
        print(f"\nSimulation Run: {self.petname} ({self.run_id})")
        print(f"Data: {self.data_path}")
        print("-"*50)

        all_params = []
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
            all_params.append(parameters_combinations)
            
        # all combinations of all lens parameters
        self.all_parameters_combinations = list(itertools.product(*all_params))
    

    def setup_simulation(self):
        # TODO: this is currently hardcoded so only height and exponent can be swept

        # generate configuration for simulations
        print(f"\nGenerating {len(self.all_parameters_combinations)} Simulation Configurations")
        self.simulation_configurations = []

        for v in tqdm(self.all_parameters_combinations):
       
            lens_combination = []
            for i, lens in enumerate(self.config["lenses"]):
                lens_dict = {
                    "name": lens["name"],
                    "height": v[i][0],   # TODO: hardcoded
                    "exponent": v[i][1], # TODO: hardcoded
                    "medium": lens["medium"]
                }
                # print(lens_dict)
                lens_combination.append(lens_dict)
            
            sim_config = {
                "run_id": self.run_id, 
                "run_petname": self.petname, 
                "log_dir": self.data_path, 
                "sim_parameters": self.config["sim_parameters"],
                "options": self.config["options"],
                "mediums": self.config["mediums"], 
                "lenses": lens_combination,
                "stages": self.config["stages"]
            }
            
            # pprint(sim_config["lenses"])
            self.simulation_configurations.append(sim_config)

            # print("-"*20)

        # save sim configurations
        utils.save_metadata(self.config, self.data_path)

    def run_simulations(self):
        print(f"\nRunning {len(self.simulation_configurations)} Simulations")

        for sim_config in tqdm(self.simulation_configurations):

            sim = Simulation.Simulation(sim_config)
            sim.run_simulation()


def generate_parameter_sweep(param: list) -> np.ndarray:
    # TODO: tests
    if isinstance(param, float):
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

    if (finish - start) < step_size:
        raise ValueError(f"Step size is larger than parameter range. {param}")

    n_steps = int((finish - start) / (step_size)) + 1 # TODO: validate this

    return np.linspace(start, finish, n_steps)