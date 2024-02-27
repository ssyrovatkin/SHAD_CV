import numpy as np
from seam_carve import compute_energy
from common import assert_ndarray_equal
from os.path import abspath, dirname, join


def test_energy():
    test_dir = dirname(abspath(__file__))
    for i, name in enumerate(["test1.npy", "test2.npy", "test3.npy"], start=1):
        img = np.load(join(test_dir, name))
        gt = np.load(join(test_dir, f"gt{i}.npy"))
        assert_ndarray_equal(actual=compute_energy(img), correct=gt)
