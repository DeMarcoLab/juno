
from pathlib import Path
from lens_simulation.Lens import Lens, LensType
from lens_simulation.Medium import Medium
from dataclasses import dataclass
import numpy as np



@dataclass
class SimulationParameters:
    A: float
    pixel_size: float 
    sim_width: float
    sim_wavelength: float
    lens_type: LensType
    padding: int = None # px

@dataclass
class SimulationOptions:
    log_dir: Path
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
    options: dict = None
    lens_inverted: bool = False
    _id: int = 0


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