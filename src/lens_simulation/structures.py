
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
    sim_height: float
    sim_wavelength: float

@dataclass
class SimulationOptions:
    log_dir: Path
    save_plot: bool = True
    debug: bool = False

@dataclass
class SimulationStage:
    lens: Lens
    output: Medium
    distances: np.ndarray = None # (propagation distances)
    lens_inverted: bool = False
    _id: int = 0
    tilt: dict = None
    wavefront: np.ndarray = None

@dataclass
class SimulationRun:
    id: str
    petname: str
    parameters: SimulationParameters
    options: SimulationOptions
    stages: list 


@dataclass
class SimulationConfig:
    beam: None
    lenses: list
    stages: list
    options: SimulationOptions
    parameters: SimulationParameters

@dataclass
class SimulationResult:
    propagation: np.ndarray = None
    sim: np.ndarray = None
    lens: Lens = None
    freq_arr: np.ndarray = None
    delta: np.ndarray = None
    phase: np.ndarray = None


@dataclass
class StageSettings:
    lens: str
    output: float
    n_steps: int
    step_size: float
    start_distance: float
    finish_distance: float
    use_equivalent_focal_distance: bool
    focal_distance_start_multiple: float
    focal_distance_multiple: float


