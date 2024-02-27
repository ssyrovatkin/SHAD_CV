import numpy as np
import pytest

from common import assert_ndarray_equal
from deconvolution import fourier_transform


def get_test_data():
    kernels = []
    shapes = []
    answers = []

    kernels.append([[1]])
    shapes.append([3, 3])
    answers.append(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )

    kernels.append(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    shapes.append([5, 5])
    answers.append(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    kernels.append(
        [
            [0, 2],
            [0, 2],
        ]
    )
    shapes.append([2, 2])
    answers.append(
        [
            [4, 4],
            [0, 0],
        ]
    )

    kernels.append(
        [
            [1, 1],
            [1, 1],
        ]
    )
    shapes.append([4, 4])
    answers.append(
        [
            [4 + 0j, 2 + 2j, 0 + 0j, 2 - 2j],
            [2 + 2j, 0 + 2j, 0 + 0j, 2 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [2 - 2j, 2 + 0j, 0 + 0j, 0 - 2j],
        ]
    )

    kernels = [np.array(kernel, dtype=np.float64) for kernel in kernels]
    shapes = [tuple(shape) for shape in shapes]
    answers = [np.array(answer, dtype=np.complex128) for answer in answers]
    assert all(shape == answer.shape for answer, shape in zip(answers, shapes))
    return kernels, shapes, answers


KERNELS, SHAPES, ANSWERS = get_test_data()


@pytest.mark.parametrize("index", range(len(ANSWERS)))
def test_fourier_transform(index):
    kernel, shape, answer = KERNELS[index], SHAPES[index], ANSWERS[index]
    assert_ndarray_equal(
        actual=fourier_transform(kernel, shape),
        correct=answer,
        rtol=0,
        atol=1e-3,
    )
