

import matplotlib.pyplot as plt
import numpy as np




def plot_simulation(arr: np.ndarray, width: int, height: int, pixel_size_x: float, start_distance: float, finish_distance: float) -> plt.Figure:
    """Plot the output simulation array.

    Args:
        arr (np.ndarray): the simulation output arrays
        width (int): [description]
        height (int): [description]
        pixel_size_x (float): [description]
        start_distance (float): [description]
        finish_distance (float): [description]

    Returns:
        [type]: [description]
    """
    min_h, max_h = arr.shape[0] // 2 - height // 2, arr.shape[0] // 2 + height // 2 
    min_w, max_w = arr.shape[1] // 2 - width // 2, arr.shape[1] // 2 + width // 2

    arr_resized = arr[min_h:max_h, min_w:max_w]

    # calculate extents (xlabel, ylabel)
    min_x = -arr_resized.shape[1] / 2 * pixel_size_x / 1e-6
    max_x = arr_resized.shape[1] / 2 * pixel_size_x / 1e-6

    # nb: these are reversed because the light comes from top...
    dist = finish_distance - start_distance
    
    min_h_frac = min_h / arr.shape[0]
    max_h_frac = max_h / arr.shape[0]

    min_y = (start_distance + max_h_frac * dist)   / 1e-3
    max_y = (start_distance + min_h_frac * dist) /  1e-3

    fig = plt.figure()
    plt.imshow(arr_resized,
               extent=[min_x,
                       max_x,
                       min_y,
                       max_y], 
                       interpolation='spline36',
               aspect='auto', cmap="jet")
    plt.title(f"Simulation Output ({width}x{height})")
    plt.ylabel("Distance (mm)")
    plt.xlabel("Distance (um)")
    plt.colorbar()
    
    return fig




def save_figure(fig, fname: str = "img.png") -> None:

    plt.savefig(fname)



