

from operator import invert
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from lens_simulation import utils
from lens_simulation.Lens import Lens, LensType
from lens_simulation.Medium import Medium

st.set_page_config(page_title="Lens Configuration", layout="wide", initial_sidebar_state="expanded")

st.title("Lens Configuration")


METRE_TO_MICRON = 1e6
MICRON_TO_METRE = 1 / METRE_TO_MICRON


with st.form("lens_form"):

    # lens params
    form_cols = st.columns(5)

    form_cols[0].subheader("Profile Parameters")
    form_cols[1].subheader("Method Parameters")
    form_cols[2].subheader("Truncation Parameters")
    form_cols[3].subheader("Grating Parameters")
    form_cols[4].subheader("Apeture Parameters")

    diameter = form_cols[0].number_input("Diameter (um)", min_value=100, max_value=10000, value=5000, step=50) * MICRON_TO_METRE
    height = form_cols[0].number_input("Height (um)", min_value=5, max_value=100, value=20, step=1) * MICRON_TO_METRE
    exponent = form_cols[0].number_input("Exponent", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    lens_type = form_cols[1].selectbox("Lens Type", [lens_type.name for lens_type in LensType])
    invert_profile = form_cols[1].selectbox("Invert Profile", [False, True])
    medium_refractive_index = form_cols[1].number_input("Refractive Index", min_value=1.0, max_value=10.0, value=1.0, step=0.01)
    
    truncation_type = form_cols[2].selectbox("Truncation Type", ["value", "radial"])
    truncation = form_cols[2].number_input("Truncation Height (um)", min_value=0.0, max_value=height * METRE_TO_MICRON, value=height*METRE_TO_MICRON, step=0.1) * MICRON_TO_METRE
    truncation_radius = form_cols[2].number_input("Truncation Radius (um)", min_value=0.0, max_value=diameter / 2 * METRE_TO_MICRON, value=diameter/2*METRE_TO_MICRON, step=1.0) * MICRON_TO_METRE

    grating_width_m = form_cols[3].number_input("Grating Width (um)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) * MICRON_TO_METRE
    grating_distance_m = form_cols[3].number_input("Grating Distance (um)", min_value=0.0, max_value=50.0 , value=0.0, step=0.1) * MICRON_TO_METRE

    inner_m = form_cols[4].number_input("Apeture Inner (um)", min_value=0.0, max_value=diameter / 2 * METRE_TO_MICRON, value=0.0, step=1.0) * MICRON_TO_METRE
    outer_m = form_cols[4].number_input("Apeture Outer (um)", min_value=0.0, max_value=diameter / 2 * METRE_TO_MICRON, value=0.0, step=1.0) * MICRON_TO_METRE

    submitted = st.form_submit_button("Generate Lens Profile")

if submitted:

    # create lens
    lens = Lens(diameter=diameter, 
                height=height, 
                exponent=exponent, 
                medium=Medium(medium_refractive_index))

    if lens_type == LensType.Spherical.name:
        lens.generate_profile(1e-6, lens_type=LensType.Spherical)

    if lens_type == LensType.Cylindrical.name:
        lens.generate_profile(1e-6, lens_type=LensType.Cylindrical)
        lens.extrude_profile(lens.diameter)

    lens.calculate_truncation_mask(truncation=truncation, radius= truncation_radius, type=truncation_type)
    lens.calculate_apeture(inner_m = inner_m, outer_m=outer_m, type="radial")
    
    if invert_profile:
        lens.invert_profile()

    # generate profile plots
    lens_fig = utils.plot_lens_profile_slices(lens, max_height=height)
    fig_2d = utils.plot_lens_profile_2D(lens)
    mask_fig = utils.plot_lens_profile_2D(lens)


    lens.calculate_grating_mask(0.5e-6, 1e-6)
    

    # show plots
    cols = st.columns(3)
    cols[0].subheader("Lens Profile Slice")
    cols[0].pyplot(lens_fig)
    cols[1].subheader("Lens Profile 2D")
    cols[1].pyplot(fig_2d)
    cols[2].subheader("Apeture Applied")
    cols[2].pyplot(mask_fig)

save_button = st.button("Save Lens Configuration")

if save_button:
    # TODO: 
    st.success("TODO: make this button do something")