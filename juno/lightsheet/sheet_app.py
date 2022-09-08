import streamlit as st


# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from juno import utils, plotting
from juno.lightsheet import light_sheet
from IPython.display import Image
import os

from pathlib import Path
from pprint import pprint

st.set_page_config(layout="wide")
st.title("Light Sheet Measurement")


width = st.number_input("crop width", 0.0, 1.0, 1.0)
height = st.number_input("crop height", 0.0, 1.0, 1.0)
centre_x = st.number_input("crop centre x", 0.0, 1.0, 0.5)
centre_y = st.number_input("crop centre y", 0.0, 1.0, 0.5)
threshold = st.number_input("threshold of max", 0.0, 1.0, 0.5)



### Sheet Measurement
@st.cache
def load_sim(path):

    return utils.load_simulation(path)
# load sim
path = st.text_input("Simulation path (.zarr)")
sim = load_sim(path)

# slice view
# front_on = plotting.slice_simulation_view(sim, axis=0, prop=0.5)
top_down = plotting.slice_simulation_view(sim, axis=1, prop=0.5)
# side_on = plotting.slice_simulation_view(sim, axis=2, prop=0.5)

# crop image
image_crop, bounds = plotting.crop_image_v3(top_down, width=width, height=height, x=centre_x, y=centre_y)

# threshold image
image_threshold = plotting.threshold_image(image_crop, threshold)

# get max point and location
mid_threshold = plotting.cross_section_image(image_threshold, axis=1, prop=0.5)
mid_image = plotting.cross_section_image(image_crop, axis=1, prop=0.5)
cz = np.argmax(mid_threshold)
cx = image_threshold.shape[1] // 2

threshold_value = np.max(mid_threshold) * threshold
max_length_px, max_width_px, (x, y) = light_sheet.calculate_sheet_size_pixels(image_threshold, threshold_value)

# print(f"sheet length: {max_length_px}px")
# print(f"max width at: {max_width_x0}, {max_width_x1}")
# print(f"max_width: {max_width_px}px at z: {max_width_z}px")

metadata = utils.load_metadata(os.path.dirname(os.path.dirname(path)))

# calculate sheet dimensions
sheet_depth, sheet_width = light_sheet.calculate_sheet_size(metadata, max_length_px, max_width_px)

st.write(f"Sheet Size: width = {sheet_width:.2e} m, depth = {sheet_depth:.2e}m")

fig_cols = st.columns(4)

# fig = plt.figure()
# plt.imshow(top_down, aspect="auto", cmap="turbo")
# plt.title("Full Top Down")
# fig_cols[0].pyplot(fig, caption="Full Propagation")


fig = plt.figure()
plt.imshow(image_crop,aspect="auto", cmap="turbo")
plt.title("Image Crop")
fig_cols[0].pyplot(fig, caption="Image Crop")

fig = plt.figure()
plt.imshow(image_threshold, aspect="auto", cmap="turbo")
plt.title("Image Threshold")
fig_cols[1].pyplot(fig, caption="Image Threshold")

fig = plt.figure()
plt.plot(mid_threshold, "b--", label="thresholded")
plt.hlines(np.max(mid_threshold) * threshold, 0, mid_threshold.shape[0], "r", linestyles="--", label="threshold value")
plt.legend(loc="best")
plt.title("Thresholded Values")
fig_cols[2].pyplot(fig, caption="Thresholded Values")


fig = plt.figure()
plt.imshow(image_threshold.astype(bool), aspect="auto", cmap="gray")
# plt.plot(cx, cz, "r+", ms=50, linewidth=20)
# plt.plot(cx, min_z, "c+", ms=50)
# plt.plot(cx, max_z, "c+", ms=50)
# plt.plot(max_width_x0, max_width_z, "m+", ms=50)
# plt.plot(max_width_x1, max_width_z, "m+", ms=50)
plt.scatter(y, x, c="cyan")
plt.title("Light Sheet")

fig_cols[3].pyplot(fig, caption="Light Sheet")