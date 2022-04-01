


from re import S
import streamlit as st

from lens_simulation import utils
import glob

import os
import json

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def show_simulation_data(sim_id, df):
    
    metadata = utils.load_metadata(sim_id)
    
    st.write("---")
    st.write(f"{metadata['petname']}: {metadata['sim_id']}")

    # # show lenses and mediums
    # TODO: show df_join data

    sim_filenames = glob.glob(os.path.join(sim_id, "*/*.npy"))

    # show simulation stages

    cols = st.columns(len(sim_filenames))

    for i, sim_stage_id in enumerate(sim_filenames):
        cols[i].write(f"Stage {i}")

        sim = utils.load_simulation(sim_stage_id)

        # faster to load the image than the sim
        img_fname = os.path.join(os.path.dirname(sim_stage_id), "img.png")
        if os.path.exists(img_fname):
            import PIL
            img = PIL.Image.open(img_fname)
            cols[i].image(img)
        else:
            fig = utils.plot_simulation(sim, sim.shape[1], sim.shape[0], pixel_size_x=metadata["sim_parameters"]["pixel_size"], start_distance=0.0, finish_distance=10.0e-3)
            cols[i].pyplot(fig)
    
    # stage metadata
    st.write(df_sim)


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

# st.subheader("DF METADATA")
# st.write(df_metadata)

st.sidebar.write("---")
filter_col = st.sidebar.selectbox("Filter Column", df_metadata.columns)
min_val, max_val = st.sidebar.slider("Select values", 
    float(df_metadata[filter_col].min()), 
    float(df_metadata[filter_col].max()), 
    (float(df_metadata[filter_col].min()), float(df_metadata[filter_col].max()))
    )


st.write(min_val, max_val)

# TODO: double slider?
df_filter = df_metadata[df_metadata[filter_col] >= min_val]
df_filter = df_filter[df_filter[filter_col] <= max_val]
st.sidebar.write(f"Filtered to {len(df_filter)} simulations")
st.subheader("Filtered Simulation Data")
st.write(df_filter)


# TODO: this could be better
filter_sim_id = df_filter["data_path"].unique()

for sim_id in filter_sim_id:
    df_sim = df_metadata[df_metadata["data_path"] == sim_id] # TODO: check if we should include the filtered out data from the same sim?
    show_simulation_data(sim_id, df_sim)






