
import imp


import matplotlib.pyplot as plt


def plot_simulation(arr, width, height):

    low, high = arr.shape[1] // 2 - width // 2, arr.shape[1] // 2 + width // 2, 

    low_height, high_height = arr.shape[0] // 2 - height // 2, arr.shape[0] // 2 + height // 2

    plt.imshow(arr[low_height:high_height, low:high+1],  aspect="auto")
    plt.show()