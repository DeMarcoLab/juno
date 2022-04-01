


from re import S
import streamlit as st

from lens_simulation import utils
import glob

import os
import json

import pandas as pd

pd.set_option("display.precision", 8)

def load_simulation_run(sim_id):

    # st.write("---")

    # delete sim
    # if st.button("DELETE THIS SIM"):
    #     st.error("SIM DELETED")

    metadata = utils.load_metadata(sim_id)

    # st.write(f"{metadata['petname']}: {metadata['sim_id']}")


    df_stages = pd.DataFrame.from_dict(metadata["stages"])
    df_stages["stage"] = df_stages.index
    # st.write(df_stages)

    df_lens = pd.DataFrame.from_dict(metadata["lenses"])
    df_lens = df_lens.rename(columns={"name": "lens"})
    # st.write(df_lens)

    df_medium = pd.DataFrame.from_dict(metadata["mediums"])
    # df_medium = df_medium.rename(columns={"name": "medium"})
    # st.write(df_medium)


    df_join = pd.merge(df_stages, df_lens, on="lens")
    # df_join = pd.merge(df_join, df_medium, on="medium")
        
    df_join["petname"] = metadata["petname"]
    df_join["sim_id"] = metadata["sim_id"]
    df_join["run_id"] = metadata["run_id"]
    df_join["data_path"] = os.path.join(metadata["log_dir"], metadata["sim_id"])
    df_join["height"] = df_join["height"] *10e3 # convert to mm
    # st.write(df_join)

    return df_join


def show_simulation_data(sim_id):
    
    metadata = utils.load_metadata(sim_id)
    
    st.subheader(f"{metadata['petname']}: {metadata['sim_id']}")

    # # show lenses and mediums
    # lens_cols = st.columns(2)
    # lens_cols[0].subheader("Lenses")

    # for lens in metadata["lenses"]:
    #     lens_cols[0].write(lens)


    # lens_cols[1].subheader("Mediums")

    # for lens in metadata["mediums"]:
    #     lens_cols[1].write(lens)

    sim_filenames = glob.glob(os.path.join(sim_id, "*/*.npy"))

    # st.write(sim_filenames)

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

run_id = st.sidebar.selectbox("Select a Run", glob.glob("log/*"))

filenames = glob.glob(os.path.join(run_id, "*"))

# selected_sim_ids = st.sidebar.multiselect("Select a simulation", filenames)

df_metadata = pd.DataFrame()

# st.write(selected_sim_ids)

# TODO: cache
for sim_id in filenames:
    df_join = load_simulation_run(sim_id)   
    df_metadata = pd.concat([df_metadata, df_join])

st.subheader("DF METADATA")
st.write(df_metadata)


filter_col = st.selectbox("Filter Column", df_metadata.columns)
filter_val = st.slider("Select values", 
    float(df_metadata[filter_col].min()), 
    float(df_metadata[filter_col].max()), 
    float(df_metadata[filter_col].min()))

# TODO: double slider?
df_filter = df_metadata[df_metadata[filter_col] > filter_val]
st.write(df_filter)


# TODO: this could be better
filter_sim_id = df_filter["data_path"].unique()

for sim_id in filter_sim_id:
    
    show_simulation_data(sim_id)






