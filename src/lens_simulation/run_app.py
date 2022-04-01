import streamlit as st

from lens_simulation import SimulationRunner, Simulation




st.title("Lens Simulation Runner")


config_filename = st.text_input("Enter the config filename", "config.yaml")

sim_runner = SimulationRunner.SimulationRunner(config_filename)
sim_runner.initialise_simulation()
sim_runner.setup_simulation()


st.write(sim_runner.simulation_configurations)
# sim_runner.run_simulations()

if st.button("Run Simulation"):

    progress_bar = st.progress(0)
    st.write("Running simulation")

    progress_steps = len(sim_runner.simulation_configurations)

    for i, sim_config in enumerate(sim_runner.simulation_configurations):

            sim = Simulation.Simulation(sim_config)
            progress_bar.progress((i+1) / progress_steps)
            # sim.run_simulation()