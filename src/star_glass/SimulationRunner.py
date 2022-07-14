import itertools
from pathlib import Path
import uuid
import os
import petname
import glob
import logging
from pprint import pprint
import numpy as np
from tqdm import tqdm

from star_glass import Simulation, utils, constants
import star_glass
from copy import deepcopy

class SimulationRunner:

    def __init__(self, config_filename: str) -> None:
        
        # load config
        self.config = utils.load_config(config_filename)
    
        # update metadata
        self.config["run_id"] =  uuid.uuid4()
        self.config["started"] = utils.current_timestamp()

        # setup logging
        self.data_path = self.create_logging_directory()
        logfile = utils.configure_logging(self.data_path)

    def create_logging_directory(self):
        
        # create name
        self.petname = self.config["options"]["name"] if self.config["options"]["name"] is not None else petname.generate(3)
        tmp_petname = self.petname

        # create logging directory
        self.data_path: Path = os.path.join(self.config["options"]["log_dir"], str(self.petname))
        
        counter = 1
        while os.path.exists(self.data_path):
            logging.info(f"A simulation of the name {os.path.basename(self.data_path)} already exists. Incrementing name...")
            tmp_petname = self.petname + f"_{counter:02d}" 
            self.data_path: Path = os.path.join(self.config["options"]["log_dir"], str(tmp_petname))
            counter+=1

        self.petname = tmp_petname
        os.makedirs(self.data_path, exist_ok=True)
        logging.info(f"A simulation run {self.petname} has been created at {self.data_path}.")

        return self.data_path


    def setup_simulation(self):

        info = {"run_id": self.config["run_id"], "run_petname": self.petname, "log_dir": self.data_path}
        self.simulation_configurations = generate_simulation_parameter_sweep(self.config, info)
        logging.info(f"Generated {len(self.simulation_configurations)} simulation configurations.")

        # save sim configurations
        utils.save_metadata(self.config, self.data_path)

    def run_simulations(self):
        logging.info(f"Run: {self.petname}")
        logging.info(f"Running {len(self.simulation_configurations)} Simulations")

        logging.info("----------- Simulation Summary -----------")
        logging.info(f"Pixel Size: {self.config['sim_parameters']['pixel_size']:.1e}m")
        logging.info(f"Number of Pixels: {int(self.config['sim_parameters']['sim_width'] / self.config['sim_parameters']['pixel_size'])} px")
        logging.info(f"Simulation Wavelength: {self.config['sim_parameters']['sim_wavelength']}m")
        logging.info(f"Simulation Size: {self.config['sim_parameters']['sim_height']:.1e}m x {self.config['sim_parameters']['sim_width']:.1e}m")
        logging.info(f"No. Stages: {len(self.config['stages']) + 1}")
        logging.info("------------------------------------------")

        progress_bar = tqdm(self.simulation_configurations)
        
        for i, sim_config in enumerate(progress_bar):
            progress_bar.set_description(f"Run {self.petname} ({i+1}/{len(progress_bar)})")
            sim = Simulation.Simulation(sim_config)
            sim.run_simulation()
            logging.info(f"Finished simulation {sim.petname}")
    
        self.finish_simulations()

    def finish_simulations(self):
        # save final sim configuration
        logging.info(f"Finished running {len(self.simulation_configurations)} Simulations")
        self.config["finished"] = utils.current_timestamp()
        utils.save_metadata(self.config, self.data_path)
        utils.save_dataframe(self.data_path)



def generate_parameter_sweep(start:float, stop:float, step_size:float) -> np.ndarray:

    if stop is None or step_size is None:
        return [start]

    if step_size == 0.0:
        raise ValueError("Step Size cannot be zero.")

    if start >= stop:
        raise ValueError(f"Start parameter {start} cannot be greater than stop parameter {stop}")

    SANTIY = +1e-12
    if (stop - start) + SANTIY < step_size:
        raise ValueError(f"Step size is larger than parameter range. {start}, {stop}, {step_size}")

    return np.arange(start, stop + SANTIY, step_size)


def sweep_config_keys(conf: dict, sweep_keys: list) -> list:
    key_params = []
    for k in sweep_keys:

        if isinstance(conf, dict) and k in conf: # check if param exists
            start, stop, step = conf[k], conf[f"{k}_stop"], conf[f"{k}_step"]
            params = generate_parameter_sweep(start, stop, step)
        else:
            params = [None]
       
        key_params.append(params)
    return key_params


def sweep_custom_config(path: Path) -> list:
    """Generate a sweep for custom lens configs based on the following rules:
        
        Rules:
            if None -> return None
            if yaml -> return only file
            if dir -> return all .yaml in folder

        Custom configs must be .yaml files and conform to the standard lens config. 
    """

    if path is None:
        custom_params = [None]
    elif os.path.isfile(path):
        custom_params = [path]
    elif os.path.isdir(path):
        custom_params = [fname for fname in glob.glob(os.path.join(path, "**/*.yaml"), recursive=True)]
    else:
        custom_params = [None]

    return [custom_params] 

def sweep_custom_profiles(path: Path) -> list:
    """Generate a sweep for custom profiles based on the following rules:
        
        Rules:
            if None -> return None
            if npy -> return only file
            if dir -> return all .npy in folder

        Profiles must be .npy files. 
    """

    if path is None:
        custom_params = [None]
    elif os.path.isfile(path):
        custom_params = [path]
    elif os.path.isdir(path):
        # files = glob.glob('files_path/[!_]*') # TODO: use proper glob file exclusion
        custom_params = [fname for fname in glob.glob(os.path.join(path, "**/*.npy"), recursive=True) if "aperture" not in fname]
    else:
        custom_params = [None]

    return [custom_params] 


def generate_lens_parameter_combinations(config) -> list:

    all_lens_params = []
    for lc in config["lenses"]:

        lp = sweep_config_keys(lc, constants.LENS_SWEEPABLE_KEYS)
        gp = sweep_config_keys(lc["grating"], constants.GRATING_SWEEPABLE_KEYS)
        tp = sweep_config_keys(lc["truncation"], constants.TRUNCATION_SWEEPABLE_KEYS)
        ap = sweep_config_keys(lc["aperture"], constants.APERTURE_SWEEPABLE_KEYS)

        # custom config sweeping...
        # cp = sweep_custom_profiles(lc[constants.CUSTOM_PROFILE_KEY])
        cp = sweep_custom_config(lc[constants.CUSTOM_CONFIG_KEY])

        lens_param = [*lp, *gp, *tp, *ap, *cp]
        parameters_combinations = list(itertools.product(*lens_param))
        all_lens_params.append(parameters_combinations)
        
    all_parameters_combinations_lens = list(itertools.product(*all_lens_params))

    return all_parameters_combinations_lens


def get_lens_configurations(lpc: list, config: dict) -> list:
    lens_combination = []
    for i, lens_config in enumerate(config["lenses"]): # careful of the order here...

        simulation_lenses = lens_config
        simulation_lenses["medium"] = lpc[i][0]   
        simulation_lenses["diameter"] = lpc[i][1]
        simulation_lenses["height"] = lpc[i][2]   
        simulation_lenses["exponent"] = lpc[i][3]

        if simulation_lenses["grating"] is not None:
            simulation_lenses["grating"]["width"] = lpc[i][4]   
            simulation_lenses["grating"]["distance"] = lpc[i][5]
            simulation_lenses["grating"]["depth"] = lpc[i][6]   
        if simulation_lenses["truncation"] is not None:
            simulation_lenses["truncation"]["height"] = lpc[i][7]   
            simulation_lenses["truncation"]["radius"] = lpc[i][8]  
        if simulation_lenses["aperture"] is not None:         
            simulation_lenses["aperture"]["inner"] = lpc[i][9]   
            simulation_lenses["aperture"]["outer"] = lpc[i][10]   
        
        simulation_lenses["custom_config"] = lpc[i][11]
    
        lens_combination.append(simulation_lenses)

    return lens_combination

def generate_beam_parameter_combinations(config: dict) -> list:

    all_beam_params = sweep_config_keys(config["beam"], constants.BEAM_SWEEPABLE_KEYS)

    all_parameters_combinations_beam = list(itertools.product(*all_beam_params))

    return all_parameters_combinations_beam


def get_beam_configurations(bpc: list, config: dict):

    simulation_beam = config["beam"]
    simulation_beam["width"] = bpc[0]
    simulation_beam["height"] = bpc[1]
    simulation_beam["position_x"] = bpc[2]
    simulation_beam["position_y"] = bpc[3]
    simulation_beam["theta"] = bpc[4]
    simulation_beam["numerical_aperture"] = bpc[5]
    simulation_beam["tilt_x"] = bpc[6]
    simulation_beam["tilt_y"] = bpc[7]
    simulation_beam["source_distance"] = bpc[8]
    simulation_beam["final_diameter"] = bpc[9]
    simulation_beam["focal_multiple"] = bpc[10]
    
    return simulation_beam

def generate_stage_parameter_combination(config: dict) -> list:

    all_stage_params = []
    for sc in config["stages"]:

        sp = sweep_config_keys(sc, constants.STAGE_SWEEPABLE_KEYS)
        
        parameters_combinations = list(itertools.product(*sp))
        all_stage_params.append(parameters_combinations)
        
    all_parameters_combinations_stage = list(itertools.product(*all_stage_params))

    return all_parameters_combinations_stage


def get_stage_configurations(spc: list, config: dict) -> list:
    
    stage_combination = []
    for i, stage_config in enumerate(config["stages"]): # careful of the order here...

        simulation_stage = stage_config
        simulation_stage["output"] = spc[i][0]   
        simulation_stage["start_distance"] = spc[i][1]
        simulation_stage["finish_distance"] = spc[i][2]   
        simulation_stage["focal_distance_start_multiple"] = spc[i][3]
        simulation_stage["focal_distance_multiple"] = spc[i][4]
    
        stage_combination.append(simulation_stage)

    return stage_combination

def generate_simulation_parameter_sweep(config: dict, info: dict = None) -> list:
    
    all_parameters_combinations_lens = generate_lens_parameter_combinations(config)
    all_parameters_combinations_beam = generate_beam_parameter_combinations(config)
    all_parameters_combinations_stage = generate_stage_parameter_combination(config)

    # create simulaton configs from parameter sweeps...

    simulation_configurations = []

    for i, bpc in enumerate(all_parameters_combinations_beam):
        beam_combination = get_beam_configurations(bpc, config)

        for j, lpc in enumerate(all_parameters_combinations_lens):
                    
            lens_combination = get_lens_configurations(lpc, config)
            
            for k, spc in enumerate(all_parameters_combinations_stage):
                stage_combination = get_stage_configurations(spc, config)


                # generate simulation config 
                sim_config = {
                    "run_id": info["run_id"], 
                    "run_petname": info["run_petname"], 
                    "log_dir": info["log_dir"], 
                    "sim_parameters": config["sim_parameters"],
                    "beam": deepcopy(beam_combination),
                    "options": config["options"], 
                    "lenses": deepcopy(lens_combination),
                    "stages": deepcopy(stage_combination)
                } 
                # NOTE: need to deepcopy due to way python dicts get updated...

                simulation_configurations.append(sim_config)

        # NOTE: be careful with the dictionary overriding

    logging.info(f"Beam Configurations: {len(all_parameters_combinations_beam)}")
    logging.info(f"Lens Configurations: {len(all_parameters_combinations_lens)}")
    logging.info(f"Stage Configurations: {len(all_parameters_combinations_stage)}")
    logging.info(f"Total Simulation Configurations: {len(simulation_configurations)}")

    return simulation_configurations
