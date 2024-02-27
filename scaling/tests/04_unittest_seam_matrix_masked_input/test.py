import numpy as np
from common import assert_ndarray_equal
from seam_carve import compute_seam_matrix


def test_seam_matrix_masked_1v():
    gt_v = np.array(
        [
            [3, 7, 7, 7, 16, 18, 26],
            [7, 9, 11, 13, 7178, 7185, 22],
            [7, 8, 13, 13, 7186, 7188, 20],
            [8, 8, 16, 14, 18, 20, 24],
        ]
    ).astype(np.float64)

    a = np.array(
        [
            [3, 4, 0, 0, 9, 2, 8],
            [7, 6, 4, 6, 3, 1, 4],
            [7, 1, 5, 2, 5, 2, 0],
            [8, 1, 8, 1, 5, 2, 4],
        ]
    ).astype(np.float64)
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.float64)
    assert_ndarray_equal(
        actual=compute_seam_matrix(a, mode="vertical", mask=mask), correct=gt_v
    )


def test_seam_matrix_masked_1h():
    gt_h = np.array(
        [
            [3, 4, 0, 0, 9, 2, 8],
            [10, 6, 4, 6, 7171, 7171, 6],
            [13, 5, 9, 6, 7179, 7176, 6],
            [13, 6, 13, 7, 11, 8, 10],
        ]
    ).astype(np.float64)

    a = np.array(
        [
            [3, 4, 0, 0, 9, 2, 8],
            [7, 6, 4, 6, 3, 1, 4],
            [7, 1, 5, 2, 5, 2, 0],
            [8, 1, 8, 1, 5, 2, 4],
        ]
    ).astype(np.float64)
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.float64)

    assert_ndarray_equal(
        actual=compute_seam_matrix(a, mode="horizontal", mask=mask), correct=gt_h
    )


def test_seam_matrix_masked_2v():
    a = np.array(
        [
            [8, 3, 8, 6, 2, 5, 4, 4, 2, 5],
            [3, 1, 4, 9, 6, 0, 3, 8, 8, 5],
            [7, 7, 7, 7, 0, 6, 0, 4, 7, 6],
            [0, 9, 6, 2, 2, 8, 3, 3, 3, 9],
            [8, 4, 2, 0, 9, 8, 6, 8, 9, 6],
            [2, 5, 8, 6, 6, 3, 3, 5, 7, 2],
            [9, 9, 4, 8, 9, 3, 6, 6, 6, 8],
            [8, 8, 9, 6, 0, 2, 3, 6, 7, 2],
            [0, 9, 5, 6, 1, 8, 5, 8, 7, 2],
            [5, 2, 4, 0, 1, 8, 1, 7, 8, 5],
        ]
    ).astype(np.float64)
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.float64)

    gt_v = np.array(
        [
            [8, 6, 12, 14, 16, 21, 20, 23, 25, 30],
            [3, 4, 8, 17, 20, 16, 19, 24, 28, 30],
            [7, 7, 25611, 25615, 25617, 26, 16, 20, 26, 28],
            [0, 9, 25610, 25608, 51208, 25625, 29, 19, 22, 31],
            [8, 4, 6, 25606, 51215, 51233, 25631, 37, 28, -25569],
            [2, 7, 12, 25612, 25625, 51225, 25618, 20, -25575, -51174],
            [9, 11, 11, 19, 25622, 25615, 15, -25582, -51176, -51168],
            [8, 8, 17, 13, 12, 9, 12, -25582, -51175, -51174],
            [0, 9, 7, 12, 7, 15, 14, -25580, -51175, -51173],
            [5, 2, 6, 6, 7, 15, 16, 21, -25572, -51170],
        ]
    ).astype(np.float64)
    assert_ndarray_equal(
        actual=compute_seam_matrix(a, mode="vertical", mask=mask), correct=gt_v
    )


def test_seam_matrix_masked_2h():
    a = np.array(
        [
            [8, 3, 8, 6, 2, 5, 4, 4, 2, 5],
            [3, 1, 4, 9, 6, 0, 3, 8, 8, 5],
            [7, 7, 7, 7, 0, 6, 0, 4, 7, 6],
            [0, 9, 6, 2, 2, 8, 3, 3, 3, 9],
            [8, 4, 2, 0, 9, 8, 6, 8, 9, 6],
            [2, 5, 8, 6, 6, 3, 3, 5, 7, 2],
            [9, 9, 4, 8, 9, 3, 6, 6, 6, 8],
            [8, 8, 9, 6, 0, 2, 3, 6, 7, 2],
            [0, 9, 5, 6, 1, 8, 5, 8, 7, 2],
            [5, 2, 4, 0, 1, 8, 1, 7, 8, 5],
        ]
    ).astype(np.float64)
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.float64)

    gt_h = np.array(
        [
            [8, 3, 8, 6, 2, 5, 4, 4, 2, 5],
            [6, 4, 7, 11, 8, 2, 7, 10, 10, 7],
            [11, 11, 25611, 25614, 25602, 8, 2, 11, 14, 13],
            [11, 20, 25617, 51204, 25610, 10, 5, 5, 14, 22],
            [19, 15, 22, 51210, 25619, 25613, 11, 13, 14, 20],
            [17, 20, 23, 25628, 51219, 25614, 14, 16, 20, 16],
            [26, 26, 24, 31, 51223, 25617, 20, -25580, -25578, 24],
            [34, 32, 33, 30, 31, 22, -25577, -51174, -51173, -25576],
            [32, 41, 35, 36, 23, -25569, -51169, -76766, -76767, -51171],
            [37, 34, 39, 23, -25568, -51161, -76765, -76760, -76759, -76762],
        ]
    ).astype(np.float64)

    assert_ndarray_equal(
        actual=compute_seam_matrix(a, mode="horizontal", mask=mask), correct=gt_h
    )
