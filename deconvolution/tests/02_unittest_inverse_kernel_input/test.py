import numpy as np
import pytest

from common import assert_ndarray_equal
from deconvolution import inverse_kernel


def get_test_data():
    kernels = []
    answers = []

    kernels.append(
        [
            [10, 1, 100],
            [1, 2, 3],
            [5, 10, 10],
        ]
    )
    answers.append(
        [
            [0.1, 0, 0.01],
            [0, 0, 0],
            [0, 0.1, 0.1],
        ]
    )

    kernels.append(
        [
            [10.0, -10.0j, 100, 2.0j],
            [2.0j, 5.0j, 0, 100.0],
            [100.0j, 1.0j, 10.0, 5.0 - 10.0j],
        ]
    )
    answers.append(
        [
            [0.1, 0.1j, 0.01, 0.0],
            [0.0, 0.0, 0.0, 0.01],
            [-0.01j, 0.0, 0.1, 0.04 + 0.08j],
        ]
    )

    kernels = [np.array(kernel, dtype=np.complex128) for kernel in kernels]
    answers = [np.array(answer, dtype=np.complex128) for answer in answers]
    return kernels, answers


KERNELS, ANSWERS = get_test_data()


@pytest.mark.parametrize("index", range(len(ANSWERS)))
def test_inverse_kernel(index):
    kernel, answer = KERNELS[index], ANSWERS[index]
    assert_ndarray_equal(
        actual=inverse_kernel(kernel, threshold=5),
        correct=answer,
        rtol=0,
        atol=1e-3,
    )
