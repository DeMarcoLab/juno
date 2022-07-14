import os
import sys

import star_glass
from star_glass import SimulationRunner
from star_glass import Simulation
from star_glass import utils

def main(config_filename):

    sim_runner = SimulationRunner.SimulationRunner(config_filename)
    sim_runner.setup_simulation()
    sim_runner.run_simulations()


def run_single_simulation(config_filename: str = "config.yaml"):

    # load config
    config = utils.load_simulation_config(config_filename)
    print("Run Name: ", config["run_id"])

    # create and run simulation
    sim = Simulation.Simulation(config)
    sim.run_simulation()

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        config_filename = sys.argv[1]
    else:
        config_filename = os.path.join(os.path.dirname(star_glass.__file__), "config.yaml")

    main(config_filename)
