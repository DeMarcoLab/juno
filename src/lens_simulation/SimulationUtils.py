
from lens_simulation.Lens import Lens, Medium, LensType
from dataclasses import dataclass
import numpy as np



@dataclass
class SimulationParameters:
    A: float
    pixel_size: float 
    sim_width: float
    sim_wavelength: float
    lens_type: LensType

@dataclass
class SimulationOptions:
    save: bool = True
    save_plot: bool = True
    verbose: bool = False
    debug: bool = False

@dataclass
class Beam:
    phase: np.ndarray

@dataclass
class SimulationStage:
    lens: Lens
    output: Medium
    n_slices: int
    start_distance: float
    finish_distance: float
    options: dict
    lens_inverted: bool = False


@dataclass
class SimulationConfig:
    config: dict

@dataclass
class SimulationRun:
    id: str
    petname: str
    config: SimulationConfig
    parameters: SimulationParameters
    options: SimulationOptions
    stages: list 