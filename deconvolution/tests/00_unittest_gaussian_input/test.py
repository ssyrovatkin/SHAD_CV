import numpy as np
import pytest

from common import assert_ndarray_equal
from deconvolution import gaussian_kernel


def get_test_data():
    sizes = []
    sigmas = []
    answers = []

    sizes.append(2)
    sigmas.append(1)
    answers.append(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )

    sizes.append(5)
    sigmas.append(1)
    answers.append(
        [
            [0.00296902, 0.0133062, 0.02193823, 0.0133062, 0.00296902],
            [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
            [0.02193823, 0.0983203, 0.16210282, 0.0983203, 0.02193823],
            [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
            [0.00296902, 0.0133062, 0.02193823, 0.0133062, 0.00296902],
        ]
    )

    core = [
        [0.00000011, 0.00033501, 0.00000011],
        [0.00033501, 0.99865950, 0.00033501],
        [0.00000011, 0.00033501, 0.00000011],
    ]
    for i in range(3):
        size = 2 * i + 3
        h_zeros = [0] * size
        v_zeros = [0] * i
        h_padded = [h_zeros] * i
        v_padded = [v_zeros + row + v_zeros for row in core]

        sizes.append(size)
        sigmas.append(1 / 4)
        answers.append(h_padded + v_padded + h_padded)

    answers = [np.array(answer, dtype=np.float64) for answer in answers]
    return sizes, sigmas, answers


SIZES, SIGMAS, ANSWERS = get_test_data()


@pytest.mark.parametrize("index", range(len(ANSWERS)))
def test_gaussian(index):
    size, sigma, answer = SIZES[index], SIGMAS[index], ANSWERS[index]
    assert_ndarray_equal(
        actual=gaussian_kernel(size=size, sigma=sigma),
        correct=answer,
        rtol=0,
        atol=1e-3,
    )
