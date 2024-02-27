import os
import numpy as np
import image_compression as ic


def test_downsampling_1():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    arr = np.arange(10*15).reshape((10, 15))
    answer = ic.downsampling(arr)
    true_answer = np.array([[73, 73, 73, 73, 74, 74, 74, 74],
       [73, 73, 73, 73, 74, 74, 74, 74],
       [73, 73, 73, 73, 74, 74, 74, 74],
       [73, 73, 73, 73, 74, 74, 74, 74],
       [73, 73, 73, 73, 74, 74, 74, 74]]).astype(np.int64)
    print(answer)
    print('@@@@@@@@@@@@@@@@@@@@@@@@')
    print(true_answer)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5


def test_downsampling_2():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    arr = np.arange(20*30).reshape((20, 30))
    answer = ic.downsampling(arr)
    true_answer = np.load('1.npy')
    print(answer)
    print('@@@@@@@@@@@@@@@@@@@@@@@@')
    print(true_answer)
    assert np.sum(np.abs(answer - true_answer)) < 1e-5
