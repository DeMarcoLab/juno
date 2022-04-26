from cmath import exp
from operator import le
import os
import glob
import PIL

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from lens_simulation import utils
from lens_simulation.Lens import Medium, Lens




def plot_lens_profile(lens: Lens):

    fig = plt.figure()
    plt.plot(lens.profile)
    plt.title(f"{lens}")
    return fig


st.set_page_config(page_title="Lens Configuration")

st.title("Lens Configuration")


METRE_TO_MICRON = 1e6
MICRON_TO_METRE = 1 / METRE_TO_MICRON


with st.form("lens_form"):

    # lens params
    form_cols = st.columns(4)
    diameter = form_cols[0].number_input("Diameter (um)", min_value=100, max_value=10000, value=5000, step=50) * MICRON_TO_METRE
    height = form_cols[1].number_input("Height (um)", min_value=5, max_value=100, value=20, step=1) * MICRON_TO_METRE
    exponent = form_cols[2].number_input("Exponent", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    medium_refractive_index = form_cols[3].number_input("Refractive Index", min_value=1.0, max_value=10.0, value=1.0, step=0.01)
    
    submitted = st.form_submit_button("Generate Lens Profile")
    if submitted:


        # create lens
        lens = Lens(diameter=diameter, 
                    height=height, 
                    exponent=exponent, 
                    medium=Medium(medium_refractive_index))
        lens.generate_profile(1e-6)


        # plot lens
        lens_fig = plot_lens_profile(lens)

        cols = st.columns(3)

        st.pyplot(lens_fig)





# extrude




# revolve