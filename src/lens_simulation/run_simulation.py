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


def run_single_simulation(config_filename):
    # load config
    conf = utils.load_config(args)

    # run_id is for when running a batch of sims, each sim has unique id
    run_id = (uuid.uuid4())
    data_path = os.path.join(os.getcwd(), "log", str(run_id))
    config = {
        "run_id": run_id,
        "log_dir": data_path,
        "sim_parameters": conf["sim_parameters"],
        "options": conf["options"],
        "mediums": conf["mediums"],
        "lenses": conf["lenses"],
        "stages": conf["stages"],
    }

    sim = Simulation.Simulation(config)
    sim.run_simulation()

if __name__ == "__main__":

    args = "config.yaml"
    if len(sys.argv) >= 2:
        args = sys.argv[1]

    main(args)
