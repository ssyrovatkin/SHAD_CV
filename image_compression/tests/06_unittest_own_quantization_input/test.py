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

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def test_own_quantization_1():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    answer = ic.own_quantization_matrix(y_quantization_matrix, 50)
    true_answer = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_own_quantization_2():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    answer = ic.own_quantization_matrix(color_quantization_matrix, 50)
    true_answer = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_own_quantization_3():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    answer = ic.own_quantization_matrix(y_quantization_matrix, 75)
    true_answer = np.array([
        [8.0, 6.0, 5.0, 8.0, 12.0, 20.0, 26.0, 31.0],
        [6.0, 6.0, 7.0, 10.0, 13.0, 29.0, 30.0, 28.0],
        [7.0, 7.0, 8.0, 12.0, 20.0, 29.0, 35.0, 28.0],
        [7.0, 9.0, 11.0, 15.0, 26.0, 44.0, 40.0, 31.0],
        [9.0, 11.0, 19.0, 28.0, 34.0, 55.0, 52.0, 39.0],
        [12.0, 18.0, 28.0, 32.0, 41.0, 52.0, 57.0, 46.0],
        [25.0, 32.0, 39.0, 44.0, 52.0, 61.0, 60.0, 51.0],
        [36.0, 46.0, 48.0, 49.0, 56.0, 50.0, 52.0, 50.0]
    ]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_own_quantization_4():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    answer = ic.own_quantization_matrix(color_quantization_matrix, 98)
    true_answer = np.array([
        [1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    ]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5
