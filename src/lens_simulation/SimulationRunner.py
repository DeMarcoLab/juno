import itertools
from pathlib import Path
import uuid
import os

import datetime
import time

from lens_simulation import Simulation, Lens

class SimulationRunner():

    def __init__(self) -> None:
        self.data_path: Path
        self.run_id = uuid.uuid4()
        self.parameters = None


    def initialise_simulation(self) -> None :

        print("Hello Sim: ", self.run_id)
        # TODO: things
    

    def setup_simulation(self):
        parameter_names = list(self.parameters.keys())
        a = [] 
        for param, values, in self.parameters.items():
            a.append(values)
        
        # get all combinations of paramters
        # ref: https://stackoverflow.com/questions/798854/all-combinations-of-a-list-of-lists
        parameters_combinations = list(itertools.product(*a))

        self.sim_parameters = [] # TODO: maybe a dictionary is better
        for params in parameters_combinations:
             self.sim_parameters.append(list(zip(parameter_names, params)))

        # create logging directory
        log_dir = os.getcwd() # TODO: make user selectable
        self.data_path = os.path.join(log_dir , "log",  str(self.run_id))
        os.makedirs(self.data_path, exist_ok=True)


    def run_simulations(self):
        print(f"Running {len(self.sim_parameters)} Simulations")

        for p in self.sim_parameters:
            d = dict(p)

            config = {
                "run_id": self.run_id,
                "log_dir": self.data_path,
                "timestaamp": datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H%M%S'),
                "parameters": d
            }
            
            sim = Simulation.Simulation(config=config)

            sim.run_simulation()
        