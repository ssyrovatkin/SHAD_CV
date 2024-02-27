import numpy as np

from common import assert_ndarray_equal
from deconvolution import compute_psnr


def test_psnr():
    img_gt = np.array(
        [
            [1, 2, 3],
            [4, 0, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )
    img_pred = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )
    correct = 43.69383
    actual = compute_psnr(img_pred, img_gt)
    assert abs(actual - correct) < 1e-3
