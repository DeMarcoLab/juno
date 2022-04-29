import os
import sys

import lens_simulation
from lens_simulation import SimulationRunner

def main(args):

    sim_runner = SimulationRunner.SimulationRunner(args)
    sim_runner.initialise_simulation()
    sim_runner.setup_simulation()
    sim_runner.run_simulations()

if __name__ == "__main__":

    args = os.path.join(os.path.dirname(lens_simulation.__file__), "config.yaml")
    if len(sys.argv) >= 2:
        args = sys.argv[1]

    main(args)
