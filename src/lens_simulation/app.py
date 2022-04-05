import streamlit as st

from lens_simulation import utils, Lens
import glob

import os
import json
import matplotlib.pyplot as plt
import PIL



import pandas as pd

pd.set_option("display.precision", 8)

@st.cache
def load_simulation_run(sim_id):

    metadata = utils.load_metadata(sim_id)

    df_stages = pd.DataFrame.from_dict(metadata["stages"])
    df_stages["stage"] = df_stages.index


    df_lens = pd.DataFrame.from_dict(metadata["lenses"])
    df_lens = df_lens.rename(columns={"name": "lens"})

    df_medium = pd.DataFrame.from_dict(metadata["mediums"])
    df_medium = df_medium.rename(columns={"name": "medium"})


    df_join = pd.merge(df_stages, df_lens, on="lens")
    df_join = pd.merge(df_join, df_medium, on="medium")

    df_join = df_join.rename(columns={"medium": "lens_medium", "output": "output_medium", "refractive_index": "lens_refractive_index"})
        
    df_join["petname"] = metadata["petname"]
    df_join["sim_id"] = metadata["sim_id"]
    df_join["run_id"] = metadata["run_id"]
    df_join["run_petname"] = metadata["run_petname"]
    df_join["sim_width"] = metadata["sim_parameters"]["sim_width"]
    df_join["pixel_size"] = metadata["sim_parameters"]["pixel_size"]
    df_join["sim_wavelength"] = metadata["sim_parameters"]["sim_wavelength"]
    df_join["data_path"] = os.path.join(metadata["log_dir"], metadata["sim_id"])
    df_join["height"] = df_join["height"] *10e3 # convert to mm

    return df_join

# TODO: make this work when sim data isnt saved...
def show_simulation_data(sim_id, df_sim):
    
    metadata = utils.load_metadata(sim_id)
    
    st.write("---")
    st.write(f"{metadata['petname']}: {metadata['sim_id']}")
    
    sim_dirnames = glob.glob(os.path.join(sim_id, "*/"))

    # show simulation stages

    cols = st.columns(len(sim_dirnames))

    for i, fname in enumerate(sim_dirnames):
        cols[i].write(f"Stage {i}")
        
        # try to load image
        img_fname = os.path.join(fname, "img.png")
        sim_fname = os.path.join(fname, "sim.npy")

        # faster to load the image than the sim
        if os.path.exists(img_fname):
            img = PIL.Image.open(img_fname)
            cols[i].image(img)
        else:
            # try to load sim
            if os.path.exists(sim_fname):
                sim = utils.load_simulation(sim_fname)
                fig = utils.plot_simulation(sim, sim.shape[1], sim.shape[0], pixel_size_x=metadata["sim_parameters"]["pixel_size"], start_distance=0.0, finish_distance=10.0e-3)
                cols[i].pyplot(fig)

        # draw lens
        df_lens = df_sim[df_sim["stage"]==i]
        lens_medium = Lens.Medium(refractive_index = float(df_lens["lens_refractive_index"]))
        lens = Lens.Lens(
            diameter=float(df_lens["sim_width"]), 
            height=float(df_lens["height"]), 
            exponent=float(df_lens["exponent"]), 
            medium=lens_medium
        )
        lens.generate_profile(pixel_size=df_lens["pixel_size"])

        fig = plt.figure()
        plt.plot(lens.profile)
        plt.title(f"{lens}")
        cols[i].pyplot(fig)

    # stage metadata
    st.write(df_sim)


st.set_page_config(page_title="Lens Simulation", layout="wide")
st.title("Lens Simulation")

petname = st.sidebar.selectbox("Select a Run", glob.glob("log/*"))

filenames = glob.glob(os.path.join(petname, "*"))

df_metadata = pd.DataFrame()

# TODO: cache
for sim_id in filenames:
    df_join = load_simulation_run(sim_id)   
    df_metadata = pd.concat([df_metadata, df_join])

st.sidebar.write("---")
filter_col = st.sidebar.selectbox("Filter Column 1", df_metadata.columns, 7)
min_val, max_val = st.sidebar.slider(f"Select values for {filter_col}", 
    float(df_metadata[filter_col].min()), 
    float(df_metadata[filter_col].max()), 
    (float(df_metadata[filter_col].min()), float(df_metadata[filter_col].max()))
    )

df_filter = df_metadata[df_metadata[filter_col] >= min_val]
df_filter = df_filter[df_filter[filter_col] <= max_val]

filter_col_2 = st.sidebar.selectbox("Filter Column 2", df_metadata.columns, 8)
values_2 = st.sidebar.multiselect(f"Select values for {filter_col_2}", 
        df_filter[filter_col_2].unique(), 
        df_filter[filter_col_2].min())
df_filter = df_filter[df_filter[filter_col].isin(values_2)]

# TODO: broke the filtering somehow, fix tomorrow

st.sidebar.write(f"Filtered to {len(df_filter['sim_id'].unique())} simulations")
st.subheader(f"Filtered Simulation Data ({petname})")
st.write(df_filter)


# TODO: filter should be multi-select instead of range? when to use which?

# TODO: this could be better
filter_sim_id = df_filter["data_path"].unique()

for sim_id in filter_sim_id:
    df_sim = df_metadata[df_metadata["data_path"] == sim_id] # TODO: check if we should include the filtered out data from the same sim?
    show_simulation_data(sim_id, df_sim)






