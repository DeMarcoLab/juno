import pytest

from lens_simulation.Lens import Lens

import numpy as np


def test_Lens():


    lens = Lens(width = 1.0, height = 1.0, exponent=2.0)

    assert lens.width == 1.0
    assert lens.height == 1.0
    assert lens.exponent == 2.0


def test_axicon_lens():

    lens = Lens(width = 1.0, height = 1.0, exponent=1.0)

    profile = lens.generate_profile()

    assert np.max(profile) == 0
    assert np.min(profile) == -lens.height
    assert profile[int(len(profile) * 0.75)] == -lens.height / 2


def test_focusing_lens():

    lens = Lens(width = 1.0, height = 1.0, exponent=2.0)

    profile = lens.generate_profile()

    assert np.max(profile) == 0
    assert np.min(profile) == -lens.height
    assert not profile[int(len(profile) * 0.75)] == -lens.height / 2