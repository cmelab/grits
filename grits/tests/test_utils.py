import numpy as np


def test_num2str():
    from grits.utils import num2str

    assert num2str(0) == "A"
    assert num2str(25) == "Z"
    assert num2str(26) == "AA"


def test_vdistance():
    from grits.utils import v_distance

    pos_array = np.zeros((2, 3))
    pos = np.array([1, 0, 0])

    assert np.array_equal(v_distance(pos, pos_array), [1, 1])

    pos_array = np.array([[1, 0, 0], [1, 1, 1]])

    assert np.allclose(v_distance(pos, pos_array), [0, 1.41421356])
