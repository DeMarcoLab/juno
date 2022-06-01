
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
class SimulationStage:
    lens: Lens
    output: Medium
    n_slices: int
    start_distance: float
    finish_distance: float
    options: dict = None
    lens_inverted: bool = False
    _id: int = 0
    tilt: dict = None


@dataclass
class SimulationRun:
    id: str
    petname: str
    parameters: SimulationParameters
    options: SimulationOptions
    stages: list 


@dataclass
class SimulationResult:
    propagation: np.ndarray = None
    top_down: np.ndarray = None
    side_on: np.ndarray = None
    sim: np.ndarray = None
    sim_profile: np.ndarray = None
    lens: Lens = None
    freq_arr: np.ndarray = None
    delta: np.ndarray = None
    phase: np.ndarray = None


