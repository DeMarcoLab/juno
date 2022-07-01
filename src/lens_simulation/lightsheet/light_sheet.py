

# N.B switch to using contours for better automation
## sheet contours?...https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
# https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/

# TODO: find longest sheet (loop through each)
import matplotlib.pyplot as plt
from lens_simulation import utils, plotting
import os
from pathlib import Path

import numpy as np


def calculate_widest_part_of_sheet_v2(image, threshold_value):

    cx = image.shape[1] // 2

    max_sheet_width_px = 0
    for j in range(image.shape[0]):
    
        for k in range(cx, image.shape[1]):
            
            if image[j, k] < threshold_value:
                width = k 

                if width > max_sheet_width_px:
                    max_sheet_width_px = width 
                    max_width_px = j
                break

    max_sheet_width_px = (max_sheet_width_px - cx) * 2 # (only half calculated)
    return max_width_px, max_sheet_width_px



def calculate_widest_part_of_sheet(image, threshold_value):
    width_dict = {}
    import collections

    max_sheet_width_px = 0
    for j in range(image.shape[0]):

        width = image[j, :] > threshold_value
        counter = collections.Counter(width)
        width_dict[j] = counter[True]
        
        if width_dict[j] > max_sheet_width_px:
            max_sheet_width_px = width_dict[j]
            max_width_px = j
    
    return max_width_px, max_sheet_width_px

def calculate_longest_sheet(above_threshold):

    max_sheet_length = 0
    current_sheet_length = 0
    for i in range(above_threshold.shape[0]):

        if above_threshold[i] == True:
            current_sheet_length += 1
        else: 
            current_sheet_length = 0
        
        if current_sheet_length > max_sheet_length:
            max_sheet_length = current_sheet_length
            max_px = i + 1
    
    return max_px, max_sheet_length


def calculate_sheet_size_pixels(image, threshold_value):

    cz = image.shape[0] // 2
    cx = image.shape[1] // 2

    min_z, max_z = cz, cz
    # find min, max values less than threshold
    for z_idx in range(cz, image.shape[0], 1):
        val = image[z_idx, cx]
        if val < threshold_value:
            max_z = z_idx
            break
            
    for z_idx in range(cz, 0, -1):
        val = image[z_idx, cx]
        if val < threshold_value:
            min_z = z_idx
            break
    
    max_length_px = max_z - min_z
    max_width_px = 0

    coords = []
    min_x_idx, max_x_idx = cx, cx
    for z_idx in range(min_z, max_z, 1):

        for x_idx in range(cx, image.shape[1], 1):
            val = image[z_idx, x_idx]
            coords.append((z_idx, x_idx))
            if val < threshold_value:
                max_x_idx = x_idx
                break

        for x_idx in range(cx, 0, -1):
            val = image[z_idx, x_idx]
            coords.append((z_idx, x_idx))
            if val < threshold_value:
                min_x_idx = x_idx
                break   

        # sheet mask

        # max width
        width = (max_x_idx - min_x_idx)

        if width > max_width_px:
            max_width_px = width
            max_width_z = z_idx
            max_width_x0 = min_x_idx
            max_width_x1 = max_x_idx

    x = [c[0] for c  in coords]
    y = [c[1] for c  in coords]

    return max_length_px, max_width_px, (x, y)



def calculate_sheet_size(metadata, max_length_px, max_width_px):

    pixel_size = (metadata["sim_parameters"]["pixel_size"])
    start_distance = metadata["stages"][0]["start_distance"] 
    finish_distance = metadata["stages"][0]["finish_distance"]
    n_steps = metadata["stages"][0]["n_steps"]
    step_size_z = (finish_distance - start_distance) / n_steps
    sheet_depth = max_length_px * step_size_z
    sheet_width = max_width_px * pixel_size

    return sheet_depth, sheet_width

### Sheet Measurement
# ASSUMPTIONS: 
# Light sheet is along the centre line
# Light sheet is the longest structure

def calculate_light_sheet_dimensions(path: Path):
    """Calculate the dimensions of a light sheet for a simulation.

    Args:
        path (Path): path to simulation file (.zarr)

    Returns:
        _type_: _description_
    """

    sim = utils.load_simulation(path)
    metadata = utils.load_metadata(os.path.dirname(os.path.dirname(path)))

    # slice view
    # front_on = plotting.slice_simulation_view(sim, axis=0, prop=0.5)
    top_down = plotting.slice_simulation_view(sim, axis=1, prop=0.5)
    # side_on = plotting.slice_simulation_view(sim, axis=2, prop=0.5)

    # crop image
    image_crop, bounds = plotting.crop_image_v3(top_down, width=1.0, height=1.0)

    # threshold image
    threshold = 0.5
    image_threshold = plotting.threshold_image(image_crop, threshold)

    mid_threshold = plotting.cross_section_image(image_threshold, axis=1, prop=0.5)
    mid_image = plotting.cross_section_image(image_crop, axis=1, prop=0.5)

    threshold_value = threshold * np.max(image_crop)

    # plt.plot(mid_threshold, "g--")
    # plt.plot(mid_image, "b--")
    # plt.hlines(threshold_value, 0, len(mid_threshold), "r", linestyles="--")
    # plt.show()


    # plt.imshow(image_threshold, aspect="auto", cmap="turbo")
    # plt.title("threshold")
    # plt.show()

    # bool mask
    above_threshold = mid_threshold > threshold_value
    max_px, max_sheet_length_px = calculate_longest_sheet(above_threshold)

    above_threshold = np.expand_dims(above_threshold, 1)
    mask = np.zeros_like(above_threshold)
    mask[max_px-max_sheet_length_px: max_px] = 1.0

    # print(len(above_threshold))

    # plt.imshow(mask, aspect="auto")
    # plt.show()

    image_longest_sheet = image_threshold[max_px-max_sheet_length_px:max_px, :]
    # plt.imshow(image_longest_sheet, aspect="auto", cmap="turbo")
    # plt.show()

    # need a way to isolate the central sheet?
    max_width_px, max_sheet_width_px = calculate_widest_part_of_sheet_v2(image_longest_sheet, threshold_value )

    cx = image_longest_sheet.shape[1]//2
    image_sheet = image_longest_sheet[:, cx-max_sheet_width_px:cx+max_sheet_width_px+1]

    sheet_depth, sheet_width = calculate_sheet_size(metadata, max_sheet_length_px, max_sheet_width_px)


    return sheet_depth, sheet_width, image_sheet
