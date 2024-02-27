import os
import numpy as np
import image_compression as ic


def test_upsampling_1():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    arr = np.arange(3*7).reshape((3, 7)).astype(np.float32)
    answer = ic.upsampling(arr)
    true_answer = np.array([
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        [7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0],
        [7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0],
        [14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 18.0, 18.0, 19.0, 19.0, 20.0, 20.0],
        [14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 18.0, 18.0, 19.0, 19.0, 20.0, 20.0]
    ]).astype(np.float64)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_upsampling_2():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    arr = np.arange(10*20).reshape((10, 20)).astype(np.float32)
    answer = ic.upsampling(arr)
    true_answer = np.load('0.npy')
    assert np.sum(np.abs(answer - true_answer)) < 1e-5
