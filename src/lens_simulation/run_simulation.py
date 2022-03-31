from lens_simulation import Simulation
import uuid
import os
import yaml
import sys

def main(args):
    with open(args, "r") as f:
        conf = yaml.full_load(f)

    run_id = uuid.uuid4()  # run_id is for when running a batch of sims, each sim has unique id
    data_path = os.path.join(os.getcwd() , "log",  str(run_id))
    config = {"run_id": run_id, 
                "log_dir": data_path, 
                "sim_parameters": conf["sim_parameters"], 
                "options": conf["options"],
                "mediums": conf["mediums"], 
                "lenses": conf["lenses"],
                "stages": conf["stages"]}

    sim = Simulation.Simulation(config)
    sim.run_simulation()

if __name__ == "__main__":

    args = "config.yaml"
    if len(sys.argv) >= 2:
        args = sys.argv[1]
            
    main(args)