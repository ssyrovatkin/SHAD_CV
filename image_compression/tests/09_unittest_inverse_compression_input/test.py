import os
import numpy as np
import image_compression as ic


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def test_inverse_compression_1():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    answer = ic.inverse_compression(ic.compression([0]))
    true_answer = np.array([0]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_inverse_compression_2():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    answer = ic.inverse_compression(ic.compression([0, 0, 0, 0, 0]))
    true_answer = np.array([0, 0, 0, 0, 0]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_inverse_compression_3():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    answer = ic.inverse_compression(ic.compression([17, 8, 54, 0, 0, 0, 97, 5, 16, 0, 45, 23, 0, 0, 0, 0, 0, 3, 67, 0, 0, 8]))
    true_answer = np.array([17, 8, 54, 0, 0, 0, 97, 5, 16, 0, 45, 23, 0, 0, 0, 0, 0, 3, 67, 0, 0, 8]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_inverse_compression_4():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    block_1 = np.array([
        [-76, -73, -67, -62, -58, -67, -64, -55],
        [-65, -69, -73, -38, -19, -43, -59, -56],
        [-66, -69, -60, -15, 16, -24, -62, -55],
        [-65, -70, -57, -6, 26, -22, -58, -59],
        [-61, -67, -60, -24, -2, -40, -60, -58],
        [-49, -63, -68, -58, -51, -60, -70, -53],
        [-43, -57, -64, -69, -73, -67, -63, -45],
        [-41, -49, -59, -60, -63, -52, -50, -34]
    ])  
    answer = ic.inverse_compression(ic.compression(ic.zigzag(ic.quantization(ic.dct(block_1), y_quantization_matrix))))
    true_answer = np.array([-26.0, -3.0, 0.0, -3.0, -2.0, -6.0, 2.0,
                            -4.0, 1.0, -3.0, 1.0, 1.0, 5.0, 1.0, 2.0,
                            -1.0, 1.0, -1.0, 2.0, -0.0, -0.0, -0.0, 0.0,
                            -0.0, -1.0, -1.0, -0.0, -0.0, 0.0, -0.0, 0.0,
                            -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0,
                            -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0,
                            -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0,
                            0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5
