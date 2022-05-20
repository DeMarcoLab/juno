import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from lens_simulation import utils
from lens_simulation.Lens import Lens, LensType, GratingSettings
from lens_simulation.Medium import Medium
from lens_simulation.beam import Beam, BeamSettings, DistanceMode, BeamSpread, BeamShape

st.set_page_config(page_title="Beam Configuration", layout="wide", initial_sidebar_state="expanded")

st.title("Beam Configuration")


METRE_TO_MICRON = 1e6
MICRON_TO_METRE = 1 / METRE_TO_MICRON


with st.form("lens_form"):

    # lens params
    form_cols = st.columns(5)

    form_cols[0].subheader("Beam Parameters")


    beam_spread = form_cols[0].selectbox("Beam Spread", [BeamSpread.Plane, BeamSpread.Converging, BeamSpread.Diverging])
    distance_mode = form_cols[0].selectbox("Distance Mode", [DistanceMode.Direct, DistanceMode.Width, DistanceMode.Focal])
    beam_shape = form_cols[0].selectbox("Beam Shape", [BeamShape.Circular, BeamShape.Square, BeamShape.Rectangular])

    width = form_cols[1].number_input("Width (um)", min_value = 1, max_value=500, value=100) * MICRON_TO_METRE
    height = form_cols[1].number_input("Height (um)", min_value = 1, max_value=150, value=50) * MICRON_TO_METRE

    pos1 = form_cols[1].number_input("Position Offset Y (um)", min_value=0, max_value=500, value=0) * MICRON_TO_METRE 
    pos2 = form_cols[1].number_input("Position Offset X (um)", min_value=0, max_value=500, value=0) * MICRON_TO_METRE

    theta = form_cols[2].number_input("Theta (deg)", min_value = 0.0, max_value=90.0, value=0.0)
    numerical_aperture = form_cols[2].number_input("Numerical Aperture (um)", min_value = 0.0, max_value=10.0, value=1.0)

    tilt = form_cols[2].number_input("Tilt (deg)", min_value = 0.0, max_value=90.0, value=0.0)

    source_distance = form_cols[3].number_input("Source Distance (um)", min_value = 1, max_value=500, value=100) * MICRON_TO_METRE
    final_width = form_cols[3].number_input("Final Beam Width (um)", min_value = 1, max_value=500, value=100) * MICRON_TO_METRE
    focal_multiple = form_cols[3].number_input("Focal Multiple (um)", min_value = 1, max_value=500, value=100) * MICRON_TO_METRE

    submitted = st.form_submit_button("Generate Beam")


    if submitted:

        beam_settings = BeamSettings(
            distance_mode = distance_mode,
            beam_spread = beam_spread,
            beam_shape = beam_shape,
            width = width,
            height = height,
            position = [pos1, pos2], # might be wrong way around
            theta = theta,
            numerical_aperture = numerical_aperture, 
            tilt = tilt,
            source_distance = source_distance,
            final_width = final_width, 
            focal_multiple = focal_multiple
        )

        st.write("Beam Settings: ", beam_settings)
        beam = Beam(beam_settings)

        st.success("Beam generated.")
        beam.generate_profile()

        fig = utils.plot_lens_profile_2D(beam.lens)
        


        fig_cols = st.columns(2)
        fig_cols[0].pyplot(fig)        


        st.success("Beam profile generated.")