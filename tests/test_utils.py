import pytest

import numpy as np
from lens_simulation import utils

def test_pad_to_equal_size():

    small = np.zeros(shape=(100, 100))  
    large = np.zeros(shape=(300, 300))
    value = 0
    padded = utils.pad_to_equal_size(small, large, value)

    assert padded.shape == large.shape, f"Padded and large should be the same shape, {padded.shape}, {large.shape}"
    assert np.allclose(padded, value), f"Padded values should be same as value, {value}"


def test_create_distance_map_px():

    w, h = 101, 101
    distance = utils.create_distance_map_px(w, h)

    # mid should be zero, min
    # edges should be max

    cx, cy = w // 2, h // 2
    min_distance = np.min(distance)
    max_distance = np.sqrt((w/2)**2 + (h/2)**2)

    assert distance[cy, cx] == min_distance == np.min(distance), "Centre point should be zero"
    assert np.allclose([distance[0, 0], max_distance], np.max(distance), atol=1), "Edges should be maximum distance"
    # assert np.allclose([distance[-1, 0], max_distance], np.max(distance), atol=1), "Edges should be maximum distance"
    # assert np.allclose([distance[0, -1], max_distance], np.max(distance), atol=1), "Edges should be maximum distance"
    # assert np.allclose([distance[-1, -1], max_distance], np.max(distance), atol=1), "Edges should be maximum distance"
    # NB: need to check this, it is true if add one more pixel to x, y


def test_calculate_num_pixels():

    # no round
    pixel_size, distance = 1.e-6, 11.e-6
    n_pixels = utils._calculate_num_of_pixels(distance, pixel_size, odd=True)
    assert n_pixels == 11

    # round up
    pixel_size, distance = 1.e-6, 11.5e-6
    n_pixels = utils._calculate_num_of_pixels(distance, pixel_size, odd=True)
    assert n_pixels == 13

    # round down
    pixel_size, distance = 1.e-6, 11.3e-6
    n_pixels = utils._calculate_num_of_pixels(distance, pixel_size, odd=True)
    assert n_pixels == 11

    # even
    pixel_size, distance = 1.e-6, 10e-6
    n_pixels = utils._calculate_num_of_pixels(distance, pixel_size, odd=False)
    assert n_pixels == 10
