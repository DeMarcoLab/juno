import pytest

import os
import numpy as np

from lens_simulation import SimulationRunner, constants, utils

@pytest.fixture
def config_with_sweep():

    config = utils.load_config(
        os.path.join(os.path.dirname(__file__), "test_config_with_sweep.yaml")
    )

    return config

## PARAMETER SWEEPS
def test_generate_parameter_sweep():

    start, stop, step = 0, 5, 1
    params = SimulationRunner.generate_parameter_sweep(start, stop, step)

    assert np.array_equal(params, [0, 1, 2, 3, 4, 5])


def test_generate_parameter_sweep_for_null_values():
    # null sweep values
    start, stop, step = 0, None, None
    params = SimulationRunner.generate_parameter_sweep(start, stop, step)
    assert np.array_equal(params, [start])

    start, stop, step = 0, 1, None
    params = SimulationRunner.generate_parameter_sweep(start, stop, step)
    assert np.array_equal(params, [start])

    start, stop, step = 0, None, 1
    params = SimulationRunner.generate_parameter_sweep(start, stop, step)
    assert np.array_equal(params, [start])


def test_generate_parameter_sweep_raises_error():

    start, stop, step, = 0, -1, 0

    # step size cannot be zero
    with pytest.raises(ValueError):
        params = SimulationRunner.generate_parameter_sweep(start, stop, step)

    start, stop, step, = 0, -1, 1

    # stop cannot be greater than start
    with pytest.raises(ValueError):
        params = SimulationRunner.generate_parameter_sweep(start, stop, step)

    start, stop, step, = 0, 1, 2

    # step size cannot be greater than range
    with pytest.raises(ValueError):
        params = SimulationRunner.generate_parameter_sweep(start, stop, step)


## BEAM PARAMETER CONFIGURATIONS
def test_generate_beam_parameter_combinations(config_with_sweep):

    config = config_with_sweep

    TOTAL_BEAM_SWEEPABLE_KEYS = len(constants.BEAM_SWEEPABLE_KEYS)
    beam_combinations = SimulationRunner.generate_beam_parameter_combinations(config)

    assert len(beam_combinations) == 3, "There should be 3 beam combinations based on the test config."

    for bc in beam_combinations:

        assert (
            len(bc) == TOTAL_BEAM_SWEEPABLE_KEYS
        ), f"Each combination should have {TOTAL_BEAM_SWEEPABLE_KEYS}, actual: {len(bc)}. {bc}"


## LENS PARAMETER CONFIGURATIONS
def test_generate_lens_parameter_combinations(config_with_sweep):
    config = config_with_sweep

    ALL_LENS_SWEEPABLE_KEYS = (*constants.LENS_SWEEPABLE_KEYS, 
                                *constants.GRATING_SWEEPABLE_KEYS, 
                                *constants.TRUNCATION_SWEEPABLE_KEYS, 
                                *constants.APERTURE_SWEEPABLE_KEYS, 
                                constants.CUSTOM_PROFILE_KEY)
    TOTAL_LENS_SWEEPABLE_KEYS = len(ALL_LENS_SWEEPABLE_KEYS)

    lens_combinations = SimulationRunner.generate_lens_parameter_combinations(config)

    assert len(lens_combinations) == 6, "There should be 6 lens combinations based on the test config."

    for lc in lens_combinations:
        for lpc in lc:
            assert (len(lpc) == TOTAL_LENS_SWEEPABLE_KEYS), f"Each combination should have {TOTAL_LENS_SWEEPABLE_KEYS}, actual: {len(lpc)}. {lpc}"


## STAGE PARAMETER CONFIGURATIONS
def test_generate_stage_parameter_combinations(config_with_sweep):
    config = config_with_sweep

    TOTAL_STAGE_SWEEPABLE_KEYS = len(constants.STAGE_SWEEPABLE_KEYS)

    stage_combinations = SimulationRunner.generate_stage_parameter_combination(config)

    assert len(stage_combinations) == 3, "There should be 3 stage combinations based on the test config."

    for sc in stage_combinations:
        for spc in sc:
            assert (
                len(spc) == TOTAL_STAGE_SWEEPABLE_KEYS
            ), f"Each combination should have {TOTAL_STAGE_SWEEPABLE_KEYS}, actual: {len(spc)}. {spc}"


def test_generate_simulation_parameter_sweep(config_with_sweep):
    
    config = config_with_sweep
    info = {"run_id": 9999, "run_petname": "test-mule", "log_dir": "test"}
    simulation_configurations = SimulationRunner.generate_simulation_parameter_sweep(config, info)
    
    assert len(simulation_configurations) == 54, f"There should be 54 total simulation configurations based on the test config, actual: {len(simulation_configurations)}: (3 beam * 6 lens * 3 stage = 54 configurations)"


## SIMULATION RUNNER
def test_simulation_runner_init():
    
    config_filename = os.path.join(os.path.dirname(__file__), "test_config.yaml")
    sim_runner = SimulationRunner.SimulationRunner(config_filename)
    sim_runner.setup_simulation()

    assert os.path.isdir(sim_runner.data_path), "Simulation Runner should create storage directory, but hasn't"
    # TODO: tear down correctly


def test_simulation_runner_with_no_sweep():

    config_filename = os.path.join(os.path.dirname(__file__), "test_config.yaml")
    sim_runner = SimulationRunner.SimulationRunner(config_filename)
    sim_runner.setup_simulation()

    assert len(sim_runner.simulation_configurations) == 1, f"There should be 54 total simulation configurations based on the test config, actual: {len(simulation_configurations)}: (3 beam * 6 lens * 3 stage = 54 configurations)"


def test_simulation_runner_with_sweep():

    config_filename = os.path.join(os.path.dirname(__file__), "test_config_with_sweep.yaml")
    sim_runner = SimulationRunner.SimulationRunner(config_filename)
    sim_runner.setup_simulation()

    assert len(sim_runner.simulation_configurations) == 54, f"There should be 54 total simulation configurations based on the test config, actual: {len(simulation_configurations)}: (3 beam * 6 lens * 3 stage = 54 configurations)"

