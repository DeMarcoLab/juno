import os
import sys
from pprint import pprint
from juno import utils as j_utils
import itertools
from juno_custom.tools import element_tools, simulation_tools, beam_tools

package_path = os.path.dirname(os.path.abspath(__file__))

MODES = ["simulation", "element"]

def main(config_filename=None):
    if config_filename is None:
        raise ValueError("Config file must be provided.")
    
    # load config and append the system information
    config = j_utils.load_yaml_config(config_filename=config_filename)

    if config.get("mode") is None:
         raise ValueError("Mode must be provided.")
    
    if config.get("mode") not in MODES:
        raise ValueError("Mode must be one of the following: {}".format(MODES))
    
    system_config = simulation_tools.load_system_config(config=config, system_config_path="system_config.yaml")
    config["system"] = system_config

    
    if config.get("mode") == "simulation":
        key_list = []
        config["lenses"] = []
        for i, element in enumerate(config["elements"]):
            element_config = j_utils.load_yaml_config(config_filename=element["path"])
            config["lenses"].append({"name": element["name"]})
            for key, value in element_config.items():
                config["lenses"][i][key] = value

            if "keys" in config["lenses"][i]:
                key_list.append(config["lenses"][i]["keys"])

        key_list = list(itertools.chain(*key_list))

        if len(key_list) > 0:
            simulation_tools.iterate_through_parameter_combinations(config=config, key_list=key_list)

    pprint(config)
    pprint(key_list)
    return

    if not (
        "keys" in config
        and config["keys"] is not None
        and isinstance(config["keys"], list)
    ):
        run(config=config)

    elif len(config["keys"]) > 0:
            simulation_tools.iterate_through_parameter_combinations(config=config)

def run(config: dict, count=0):
    
    if config.get("mode") == "simulation":
        for element in config["elements"]:

            

            config = beam_tools.generate_beam_config(config=config)
            config = simulation_tools.generate_simulation_config(config=config)
    
        
    element = element_tools.generate_element(config=config)
    element.generate_profile()
    
    simulation_tools.save_outputs(config=config, profile=element.profile, count=count)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        raise ValueError("Config file must be provided.")
