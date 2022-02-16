import pytest

from lens_simulation.Lens import Lens, Medium

import numpy as np


def test_Lens():

    lens = Lens(diameter=1.0, height=1.0, exponent=2.0)

    assert lens.diameter == 1.0
    assert lens.height == 1.0
    assert lens.exponent == 2.0


def test_axicon_lens():

    lens = Lens(diameter=1.0, height=1.0, exponent=1.0)

    profile = lens.generate_profile()

    assert np.isclose(np.round(np.min(profile), 7), 0)  # TODO: broken
    assert np.isclose(np.max(profile), lens.height)
    assert profile[int(len(profile) * 0.75)] == lens.height / 2


def test_focusing_lens():

    lens = Lens(diameter=1.0, height=1.0, exponent=2.0)

    profile = lens.generate_profile()

    print(np.min(profile))
    print(np.max(profile))

    # TODO: check here on the tolerance values. i.e. FIB, print resolution

    assert np.isclose(0, np.min(profile), atol=100e-9)
    assert np.isclose(np.max(profile), lens.height, rtol=10e-6)
    assert not profile[int(len(profile) * 0.75)] == -lens.height / 2
