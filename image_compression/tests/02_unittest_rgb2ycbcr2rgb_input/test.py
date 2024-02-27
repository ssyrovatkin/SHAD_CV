import os
import numpy as np
import image_compression as ic


def test_rgb2ycbcr():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    img = np.arange(5*4*3).reshape((5, 4, 3)).astype(np.float32)
    answer = ic.rgb2ycbcr(img)
    true_answer = np.array([[[  0, 128, 127],
        [  3, 128, 127],
        [  6, 128, 127],
        [  9, 128, 127]],

       [[ 12, 128, 127],
        [ 15, 128, 127],
        [ 18, 128, 127],
        [ 21, 128, 127]],

       [[ 24, 128, 127],
        [ 27, 128, 127],
        [ 30, 128, 127],
        [ 33, 128, 127]],

       [[ 36, 128, 127],
        [ 39, 128, 127],
        [ 42, 128, 127],
        [ 45, 128, 127]],

       [[ 48, 128, 127],
        [ 51, 128, 127],
        [ 54, 128, 127],
        [ 57, 128, 127]]], dtype=np.uint8)
    assert np.max(np.abs(answer - true_answer)) <= 2


def test_ycbcr2rgb():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    img = np.arange(5*4*3).reshape((5, 4, 3)).astype(np.float32)
    answer = ic.ycbcr2rgb(ic.rgb2ycbcr(img))
    true_answer = np.array([[[ 0,  0,  0],
        [ 1,  3,  3],
        [ 4,  6,  6],
        [ 7,  9,  9]],

       [[10, 12, 12],
        [13, 15, 15],
        [16, 18, 18],
        [19, 21, 21]],

       [[22, 24, 24],
        [25, 27, 27],
        [28, 30, 30],
        [31, 33, 33]],

       [[34, 36, 36],
        [37, 39, 39],
        [40, 42, 42],
        [43, 45, 45]],

       [[46, 48, 48],
        [49, 51, 51],
        [52, 54, 54],
        [55, 57, 57]]], dtype=np.uint8)
    assert np.max(np.abs(answer - true_answer)) <= 3
