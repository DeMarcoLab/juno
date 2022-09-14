import os
import sys

import juno
from juno import SimulationRunner

def main(config_filename):

    if len(sys.argv) >= 2:
        config_filename = sys.argv[1]
    else:
        config_filename = os.path.join(os.path.dirname(juno.__file__), "config.yaml")

    sim_runner = SimulationRunner.SimulationRunner(config_filename)
    sim_runner.setup_simulation()
    sim_runner.run_simulations()

if __name__ == "__main__":

    main()
