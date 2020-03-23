
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from prmlmy.preprocess import IndexEncoder


def test_IndexEncoder():
    encoder = IndexEncoder()
    encoder.fit(["c", "b", "d", "b"])
    x = ["c", "b", "d", "b"]
    y = [1, 0, 2, 0]
    assert_array_equal(encoder.transform(x), y)
    assert_array_equal(encoder.reverse_transform(y), x)
