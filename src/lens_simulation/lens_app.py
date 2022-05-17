

from operator import invert
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from lens_simulation import utils
from lens_simulation.Lens import Lens, LensType, GratingSettings
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
    form_cols[2].subheader("Grating Parameters")
    form_cols[3].subheader("Truncation Parameters")
    form_cols[4].subheader("Apeture Parameters")

    diameter = form_cols[0].number_input("Diameter (um)", min_value=100, max_value=10000, value=1000, step=50) * MICRON_TO_METRE
    height = form_cols[0].number_input("Height (um)", min_value=5, max_value=100, value=20, step=1) * MICRON_TO_METRE
    exponent = form_cols[0].number_input("Exponent", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    lens_name = form_cols[0].text_input("Lens Name", "lens_x")
    
    lens_type = form_cols[1].selectbox("Lens Type", [lens_type.name for lens_type in LensType])
    invert_profile = form_cols[1].selectbox("Invert Profile", [False, True])
    medium_refractive_index = form_cols[1].number_input("Refractive Index", min_value=1.0, max_value=10.0, value=1.0, step=0.01)
    medium_name = form_cols[1].text_input("Medium Name", "medium_x")

    grating_width_m = form_cols[2].number_input("Grating Width (um)", min_value=0.0, max_value=1000.0, value=50.0, step=0.1) * MICRON_TO_METRE
    grating_distance_m = form_cols[2].number_input("Grating Distance (um)", min_value=0.0, max_value=diameter * METRE_TO_MICRON , value=diameter*METRE_TO_MICRON / 2, step=0.1) * MICRON_TO_METRE
    grating_depth_m = form_cols[2].number_input("Grating Depth (um)", min_value=0.0, max_value=height * METRE_TO_MICRON , value=0.0, step=0.1) * MICRON_TO_METRE
    grating_x = form_cols[2].selectbox("Grating Direction X", [True, False])
    grating_y = form_cols[2].selectbox("Grating Direction Y", [True, False])
    grating_centred = form_cols[2].selectbox("Grating Centred", [True, False])


    truncation_type = form_cols[3].selectbox("Truncation Type", ["value", "radial"])
    truncation = form_cols[3].number_input("Truncation Height (um)", min_value=0.0, max_value=height * METRE_TO_MICRON, value=height*METRE_TO_MICRON, step=0.1) * MICRON_TO_METRE
    truncation_radius = form_cols[3].number_input("Truncation Radius (um)", min_value=0.0, max_value=diameter / 2 * METRE_TO_MICRON, value=0.0, step=1.0) * MICRON_TO_METRE

    inner_m = form_cols[4].number_input("Apeture Inner (um)", min_value=0.0, max_value=diameter / 2 * METRE_TO_MICRON, value=0.0, step=1.0) * MICRON_TO_METRE
    outer_m = form_cols[4].number_input("Apeture Outer (um)", min_value=0.0, max_value=diameter / 2 * METRE_TO_MICRON, value=0.0, step=1.0) * MICRON_TO_METRE
    apeture_type = form_cols[4].selectbox("Apeture Type", ["radial", "square"])
    invert_apeture = form_cols[4].selectbox("Invert Apeture", [False, True])

    submitted = st.form_submit_button("Generate Lens Profile")

if submitted:

    # settings
    grating_settings = GratingSettings(
        width = grating_width_m,
        distance = grating_distance_m,
        depth = grating_depth_m,
        centred = grating_centred
    )


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

    
    if invert_profile:
        lens.invert_profile()

    # generate profile plots
    lens_fig = utils.plot_lens_profile_slices(lens, max_height=height)
    lens2d_fig = utils.plot_lens_profile_2D(lens)
    
    lens.generate_profile(1e-6, lens_type=LensType.Spherical)
    lens.calculate_grating_mask(grating_settings, x_axis=grating_x, y_axis=grating_y)
    lens.apply_masks(grating=True)
    grating_fig = utils.plot_lens_profile_2D(lens)
    grating1d_fig = utils.plot_lens_profile_slices(lens, max_height=height)

    lens.generate_profile(1e-6, lens_type=LensType.Spherical)
    lens.calculate_truncation_mask(truncation=truncation, radius= truncation_radius, type=truncation_type)
    lens.apply_masks(truncation=True)
    truncation_fig = utils.plot_lens_profile_2D(lens)
    truncation1d_fig = utils.plot_lens_profile_slices(lens, max_height=height)

    lens.generate_profile(1e-6, lens_type=LensType.Spherical)
    lens.calculate_apeture(inner_m = inner_m, outer_m=outer_m, type=apeture_type, inverted=invert_apeture) 
    lens.apply_masks(apeture=True)
    apeture_fig = utils.plot_lens_profile_2D(lens)
    apeture1d_fig = utils.plot_lens_profile_slices(lens, max_height=height)


    lens.generate_profile(1e-6, lens_type=LensType.Spherical)
    lens.calculate_grating_mask(grating_settings, x_axis=grating_x, y_axis=grating_y)
    lens.calculate_truncation_mask(truncation=truncation, radius= truncation_radius, type=truncation_type)
    lens.calculate_apeture(inner_m = inner_m, outer_m=outer_m, type=apeture_type)
    lens.apply_masks(grating=True, truncation=True, apeture=True) # TODO: inverting doesnt work in this case?
    mask_fig = utils.plot_lens_profile_2D(lens)
    mask1d_fig = utils.plot_lens_profile_slices(lens, max_height=height)  

    # show plots
    cols = st.columns(5)
    cols[0].subheader("Lens Profile")
    cols[0].pyplot(lens_fig)
    cols[0].pyplot(lens2d_fig)

    cols[1].subheader("Grating Profile")
    cols[1].pyplot(grating1d_fig)
    cols[1].pyplot(grating_fig)
    
    cols[2].subheader("Truncation Profile")
    cols[2].pyplot(truncation1d_fig)
    cols[2].pyplot(truncation_fig)
    
    cols[3].subheader("Apeture Applied")
    cols[3].pyplot(apeture1d_fig)
    cols[3].pyplot(apeture_fig)

    cols[4].subheader("All Masks")
    cols[4].pyplot(mask1d_fig)
    cols[4].pyplot(mask_fig)

save_button = st.button("Save Lens Configuration")

if save_button:
    # TODO: 

    lens_config = {
        "name": lens_name,
        "diameter" : diameter,
        "height" : height,
        "exponent" : exponent, 
        "medium" : medium_name,
        "custom" : None,
        "grating" : {
            "width" : grating_width_m,
            "distance" : grating_distance_m,
            "depth" : grating_depth_m,
            "x" : grating_x,
            "y" : grating_y,
            "centred" : grating_centred
        },
        "truncation": {
            "type": truncation_type,
            "height": truncation,
            "radius": truncation_radius,
        },
        "apeture": {
            "type": apeture_type,
            "inner": inner_m,
            "outer": outer_m,
            "invert": invert_apeture
        }
    }    

    import yaml
    fname = "lens.yaml"
    with open(fname, 'w') as f:
        yaml.dump(lens_config, f, sort_keys=False)

        st.success(f"Lens configuration save to {fname}")
