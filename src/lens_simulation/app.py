import os
import glob
import PIL

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from lens_simulation import utils, Lens

pd.set_option("display.precision", 8)

# @st.cache # TODO: fix cache issue
def load_data(filenames):
    df_metadata = pd.DataFrame()

    for sim_path in filenames:
        df_join = load_simulation_run(sim_path)
        df_metadata = pd.concat([df_metadata, df_join])

    return df_metadata


@st.cache
def load_simulation_run(sim_path):

    metadata = utils.load_metadata(sim_path)

    df_stages = pd.DataFrame.from_dict(metadata["stages"])
    df_stages["stage"] = df_stages.index

    df_lens = pd.DataFrame.from_dict(metadata["lenses"])
    df_lens = df_lens.rename(columns={"name": "lens"})

    df_medium = pd.DataFrame.from_dict(metadata["mediums"])
    df_medium = df_medium.rename(columns={"name": "medium"})

    # merge (join) dataframes
    df_join = pd.merge(df_stages, df_lens, on="lens")
    df_join = pd.merge(df_join, df_medium, on="medium")

    df_join = df_join.rename(
        columns={
            "medium": "lens_medium",
            "output": "output_medium",
            "refractive_index": "lens_refractive_index",
            "height": "lens_height",
            "exponent": "lens_exponent",
        }
    )  # TODO: do this with prefix on merge instead...

    df_join["petname"] = metadata["petname"]
    df_join["sim_id"] = metadata["sim_id"]
    df_join["run_id"] = metadata["run_id"]
    df_join["run_petname"] = metadata["run_petname"]
    df_join["sim_width"] = metadata["sim_parameters"]["sim_width"]
    df_join["pixel_size"] = metadata["sim_parameters"]["pixel_size"]
    df_join["sim_wavelength"] = metadata["sim_parameters"]["sim_wavelength"]
    df_join["data_path"] = os.path.join(metadata["log_dir"], metadata["sim_id"])
    df_join["lens_height"] = df_join["lens_height"]  # convert to mm
    df_join["start_distance"] = round(df_join["start_distance"], 3)
    df_join["finish_distance"] = round(df_join["finish_distance"], 3)

    # df_join["lens_inverted"] = True if df_join["lens_inverted"] == "true" else False

    return df_join


def plot_lens_profile(df, stage_no):
    df_lens = df[df["stage"] == stage_no]
    lens_medium = Lens.Medium(refractive_index=float(df_lens["lens_refractive_index"]))
    lens = Lens.Lens(
        diameter=float(df_lens["sim_width"]),
        height=float(df_lens["lens_height"]),
        exponent=float(df_lens["lens_exponent"]),
        medium=lens_medium,
    )
    lens.generate_profile(pixel_size=df_lens["pixel_size"])

    # invert the profile
    if bool(df_lens["lens_inverted"].values[0]) is True:
        lens.invert_profile()

    fig = plt.figure()
    plt.plot(lens.profile)
    plt.title(f"{lens}")
    return fig


@st.cache
def filter_dataframe(df: pd.DataFrame, filter_col: str, filter_values: list):
    """Filter a dataframe based on column values"""
    return df[df[filter_col].isin(filter_values)]


def select_filter_data(df, n_filters):
    filter_cols = st.columns(2)
    df_filter = df
    for i in range(n_filters):

        filter_col = filter_cols[0].selectbox(f"Filter Column {i+1}", df.columns, i)
        filter_values = filter_cols[1].multiselect(
            f"Select values for {filter_col}",
            df_filter[filter_col].unique(),
            df_filter[filter_col].min(),
        )

        df_filter = filter_dataframe(df_filter, filter_col, filter_values)

    return df_filter


def show_simulation_data(sim_path, df_sim):
    """Show the simulation configuration, lens profiles and results"""
    metadata = utils.load_metadata(sim_path)

    st.write("---")
    st.subheader(f"{metadata['petname']} ({metadata['sim_id']})")

    # simulation stage metadata
    st.write("Configuration")
    st.write(df_sim)

    # show simulation stages
    sim_dirnames = glob.glob(os.path.join(sim_path, "*/"))
    cols = st.columns(len(sim_dirnames))

    for i, fname in enumerate(sim_dirnames):
        cols[i].write(f"Stage {i}")

        # try to load image
        img_fname = os.path.join(fname, "topdown.png")
        sideon_fname = os.path.join(fname, "sideon.png")
        freq_fname = os.path.join(fname, "freq.png")
        delta_fname = os.path.join(fname, "delta.png")
        phase_fname = os.path.join(fname, "phase.png")
        sim_fname = os.path.join(fname, "sim.npy")

        # plot lens profile
        fig = plot_lens_profile(df=df_sim, stage_no=i)
        cols[i].pyplot(fig)
        plt.close(fig)

        # plot sim result
        if os.path.exists(img_fname):
            # faster to load the image than the sim
            img = PIL.Image.open(img_fname)
            cols[i].image(img)
            if os.path.exists(sideon_fname):
                simg = PIL.Image.open(sideon_fname)
                cols[i].image(simg)
            if os.path.exists(freq_fname):
                fimg = PIL.Image.open(freq_fname)
                cols[i].image(fimg)
            if os.path.exists(delta_fname):
                dimg = PIL.Image.open(delta_fname)
                cols[i].image(dimg)
            if os.path.exists(phase_fname):
                pimg = PIL.Image.open(phase_fname)
                cols[i].image(pimg)
        else:
            # try to load sim
            if os.path.exists(sim_fname):
                sim = utils.load_simulation(sim_fname)
                fig = utils.plot_simulation(
                    sim,
                    sim.shape[1],
                    sim.shape[0],
                    pixel_size_x=metadata["sim_parameters"]["pixel_size"],
                    start_distance=metadata["stages"][i]["start_distance"],
                    finish_distance=metadata["stages"][i]["finish_distance"],
                )
                cols[i].pyplot(fig)
                plt.close(fig)


def main():
    st.set_page_config(page_title="Lens Simulation", layout="wide")
    st.sidebar.title("Lens Simulation")

    # load data
    petname = st.sidebar.selectbox("Select a Run", glob.glob("log/*"))
    filenames = glob.glob(os.path.join(petname, "*/"))
    df_metadata = load_data(filenames=filenames)
    st.sidebar.write(f"{len(df_metadata['sim_id'].unique())} simulations loaded.")

    # visualisation options
    st.sidebar.subheader("Options")
    n_filters = st.sidebar.number_input("Number of filters", 1, 5)

    # filter simulation data
    # TODO: convert the filter to a form?
    st.write("---")
    st.subheader("Filter Simulation Data")

    df_filter = select_filter_data(df_metadata, n_filters=n_filters)

    st.write(f"Filtered to {len(df_filter['sim_id'].unique())} simulations.")
    st.write("---")
    st.header(f"Simulation Data ({petname})")
    st.write(df_filter)

    # show simulation data
    # TODO: this could be better (refactor)
    filtered_sim_paths = df_filter["data_path"].unique()

    for sim_path in filtered_sim_paths:
        df_sim = df_metadata[df_metadata["data_path"] == sim_path]
        show_simulation_data(sim_path, df_sim)


if __name__ == "__main__":
    main()

