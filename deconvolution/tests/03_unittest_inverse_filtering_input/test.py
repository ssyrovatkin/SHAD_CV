import os

import numpy as np
import pytest

from common import assert_ndarray_equal
from deconvolution import inverse_filtering


@pytest.mark.parametrize("img_size", [3, 5, 8])
@pytest.mark.parametrize("kernel_size", [1, 3])
def test_inverse_filtering_identity(img_size, kernel_size):
    img = 1 + np.arange(img_size * img_size)
    img = img.reshape(img_size, img_size).astype(np.float64)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1
    threshold = 0

    assert_ndarray_equal(
        actual=inverse_filtering(img.copy(), kernel, threshold),
        correct=img,
        rtol=0,
        atol=1e-3,
    )


def test_inverse_filtering_image():
    dirname = os.path.dirname(__file__)
    original_img = np.load(os.path.join(dirname, "original_img.npy"))
    blurred_img = np.load(os.path.join(dirname, "blurred_img.npy"))
    kernel = np.load(os.path.join(dirname, "kernel.npy"))
    threshold = 1e-10

    restored_img = inverse_filtering(blurred_img, kernel, threshold)
    mse = np.mean((restored_img - original_img) ** 2)
    assert mse < 1e-3
