import numpy as np
from common import assert_ndarray_equal
from os.path import abspath, join, dirname
from seam_carve import remove_minimal_seam


def test_remove_minimal_seam_1v():
    a_v = np.array(
        [
            [3, 7, 7, 7, 16, 12, 19],
            [7, 9, 11, 13, 10, 11, 15],
            [7, 8, 13, 13, 18, 12, 11],
            [8, 8, 16, 14, 18, 20, 16],
        ],
        dtype=np.float64,
    )
    a = np.arange(a_v.shape[0] * a_v.shape[1] * 3).reshape((*a_v.shape, 3))

    gt_v_seam = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    gt_v = np.array(
        [
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
                [30, 31, 32],
                [12, 13, 14],
                [15, 16, 17],
                [18, 19, 20],
            ],
            [
                [42, 43, 44],
                [45, 46, 47],
                [48, 49, 50],
                [51, 52, 53],
                [54, 55, 56],
                [57, 58, 59],
                [39, 40, 41],
            ],
            [
                [63, 64, 65],
                [66, 67, 68],
                [69, 70, 71],
                [72, 73, 74],
                [75, 76, 77],
                [78, 79, 80],
                [81, 82, 83],
            ],
        ],
        dtype=np.uint8,
    )

    img, _, seam_mask = remove_minimal_seam(a, a_v, mode="vertical shrink")
    assert_ndarray_equal(actual=img, correct=gt_v)
    assert_ndarray_equal(actual=seam_mask, correct=gt_v_seam)


def test_remove_minimal_seam_1h():
    a_h = np.array(
        [
            [3, 4, 0, 0, 9, 2, 8],
            [10, 6, 4, 6, 3, 3, 6],
            [13, 5, 9, 5, 8, 5, 3],
            [13, 6, 13, 6, 10, 5, 7],
        ],
        dtype=np.float64,
    )

    a = np.arange(a_h.shape[0] * a_h.shape[1] * 3).reshape((*a_h.shape, 3)) % 256

    gt_h_seam = np.array(
        [
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )

    gt_h = np.array(
        [
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [18, 19, 20]],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
                [30, 31, 32],
                [33, 34, 35],
                [39, 40, 41],
            ],
            [
                [42, 43, 44],
                [45, 46, 47],
                [48, 49, 50],
                [51, 52, 53],
                [54, 55, 56],
                [57, 58, 59],
            ],
            [
                [63, 64, 65],
                [66, 67, 68],
                [69, 70, 71],
                [72, 73, 74],
                [75, 76, 77],
                [81, 82, 83],
            ],
        ],
        dtype=np.uint8,
    )

    img, _, seam_mask = remove_minimal_seam(a, a_h, mode="horizontal shrink")
    assert_ndarray_equal(actual=img, correct=gt_h)
    assert_ndarray_equal(actual=seam_mask, correct=gt_h_seam)


def test_remove_minimal_seam_2v():
    a_v = np.array(
        [
            [8, 6, 12, 14, 16, 21, 12, 15, 17, 22],
            [3, 4, 8, 17, 20, 8, 11, 16, 20, 22],
            [7, 7, 11, 15, 8, 14, 8, 12, 18, 20],
            [0, 9, 10, 8, 8, 16, 17, 11, 14, 23],
            [8, 4, 6, 6, 15, 16, 21, 25, 20, 20],
            [2, 7, 12, 12, 12, 15, 18, 20, 25, 22],
            [9, 11, 11, 19, 21, 15, 15, 18, 24, 32],
            [8, 8, 17, 13, 12, 9, 12, 18, 25, 26],
            [0, 9, 7, 12, 7, 15, 14, 20, 25, 27],
            [5, 2, 6, 6, 7, 15, 16, 21, 28, 30],
        ],
        dtype=np.float64,
    )
    a = np.arange(a_v.shape[0] * a_v.shape[1] * 3).reshape((*a_v.shape, 3)) % 256

    test_dir = dirname(abspath(__file__))
    gt_v = np.load(join(test_dir, "gt_v.npy"))

    gt_v_seam = np.load(join(test_dir, "gt_v_seam.npy"))

    img, _, seam_mask = remove_minimal_seam(a, a_v, mode="vertical shrink")
    assert_ndarray_equal(actual=img, correct=gt_v)
    assert_ndarray_equal(actual=seam_mask, correct=gt_v_seam)


def test_remove_minimal_seam_2h():
    a_h = np.array(
        [
            [8, 3, 8, 6, 2, 5, 4, 4, 2, 5],
            [6, 4, 7, 11, 8, 2, 7, 10, 10, 7],
            [11, 11, 11, 14, 2, 8, 2, 11, 14, 13],
            [11, 20, 17, 4, 4, 10, 5, 5, 14, 22],
            [19, 15, 6, 4, 13, 12, 11, 13, 14, 20],
            [17, 11, 12, 10, 10, 14, 14, 16, 20, 16],
            [20, 20, 14, 18, 19, 13, 20, 20, 22, 24],
            [28, 22, 23, 20, 13, 15, 16, 26, 27, 24],
            [22, 31, 25, 19, 14, 21, 20, 24, 31, 26],
            [27, 24, 23, 14, 15, 22, 21, 27, 32, 31],
        ],
        dtype=np.float64,
    )
    a = np.arange(a_h.shape[0] * a_h.shape[1] * 3).reshape((*a_h.shape, 3)) % 256

    test_dir = dirname(abspath(__file__))
    gt_h = np.load(join(test_dir, "gt_h.npy"))

    gt_h_seam = np.load(join(test_dir, "gt_h_seam.npy"))

    img, _, seam_mask = remove_minimal_seam(a, a_h, mode="horizontal shrink")
    assert_ndarray_equal(actual=img, correct=gt_h)
    assert_ndarray_equal(actual=seam_mask, correct=gt_h_seam)
