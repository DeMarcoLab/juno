


from re import S
import streamlit as st

from lens_simulation import utils
import glob

import os
import json

def load_simulation_run(sim_id):

    st.write("---")

    # delete sim
    # if st.button("DELETE THIS SIM"):
    #     st.error("SIM DELETED")

    metadata = utils.load_metadata(sim_id)

    st.header(f"{metadata['petname']}: {metadata['sim_id']}")


    # show lenses and mediums
    lens_cols = st.columns(2)
    lens_cols[0].subheader("Lenses")

    for lens in metadata["lenses"]:
        lens_cols[0].write(lens)


    lens_cols[1].subheader("Mediums")

    for lens in metadata["mediums"]:
        lens_cols[1].write(lens)

    sim_filenames = glob.glob(os.path.join(sim_id, "*/*.npy"))

    # show simulation stages
    st.write("---")
    st.subheader("All Simulation Stages")


    cols = st.columns(len(sim_filenames))

    for i, sim_stage_id in enumerate(sim_filenames):
        cols[i].write(sim_stage_id)

        sim = utils.load_simulation(sim_stage_id)

        cols[i].write(sim.shape)

        fig = utils.plot_simulation(sim, sim.shape[1], sim.shape[0], pixel_size_x=metadata["sim_parameters"]["pixel_size"], start_distance=0.0, finish_distance=10.0e-3)
        cols[i].pyplot(fig)

        stage_metadata = metadata["stages"][i]
        cols[i].write(stage_metadata)







st.set_page_config(page_title="Lens Simulation", layout="wide")
st.title("Lens Simulation")

filenames = glob.glob("log/**/*")

sim_ids = st.sidebar.multiselect("Select a simulation", filenames)


for sim_id in sim_ids:
    load_simulation_run(sim_id)




