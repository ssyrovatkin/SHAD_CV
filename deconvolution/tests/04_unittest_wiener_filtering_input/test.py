import os

import numpy as np
import pytest

from common import assert_ndarray_equal
from deconvolution import wiener_filtering


def get_test_data():
    images = []
    kernels = []
    answers = []

    images.append(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    kernels.append([[1]])
    answers.append(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    images.append(
        [
            [3.4, 3.8, 4.2],
            [4.6, 5.0, 5.4],
            [5.8, 6.2, 6.6],
        ]
    )
    kernels.append(
        [
            [0.0, 0.2, 0.0],
            [0.2, 0.2, 0.2],
            [0.0, 0.2, 0.0],
        ]
    )
    answers.append(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    images = [np.array(image, dtype=np.float64) for image in images]
    kernels = [np.array(kernel, dtype=np.float64) for kernel in kernels]
    answers = [np.array(answer, dtype=np.float64) for answer in answers]
    return images, kernels, answers


IMAGES, KERNELS, ANSWERS = get_test_data()


@pytest.mark.parametrize("index", range(len(ANSWERS)))
def test_wiener_filtering_simple(index):
    blurred_img, kernel, restored_img = IMAGES[index], KERNELS[index], ANSWERS[index]
    assert_ndarray_equal(
        actual=wiener_filtering(blurred_img, kernel, K=0),
        correct=restored_img,
        rtol=0,
        atol=1e-3,
    )


def test_wiener_filtering_image():
    dirname = os.path.dirname(__file__)
    original_img = np.load(os.path.join(dirname, "original_img.npy"))
    blurred_img = np.load(os.path.join(dirname, "blurred_img.npy"))
    kernel = np.load(os.path.join(dirname, "kernel.npy"))
    K = 3e-4

    restored_img = wiener_filtering(blurred_img, kernel, K)
    mse = np.mean((restored_img - original_img) ** 2)
    assert mse < 1e-2
