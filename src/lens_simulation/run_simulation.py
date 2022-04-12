from lens_simulation import Simulation, SimulationRunner, utils
import uuid
import os
import yaml
import sys


def main(args):

    sim_runner = SimulationRunner.SimulationRunner(args)
    sim_runner.initialise_simulation()
    sim_runner.setup_simulation()
    sim_runner.run_simulations()

if __name__ == "__main__":

    args = "config.yaml"
    if len(sys.argv) >= 2:
        args = sys.argv[1]

    main(args)
