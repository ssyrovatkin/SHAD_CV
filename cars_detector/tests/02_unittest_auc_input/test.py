import time

import numpy as np
from detection_and_metrics import calc_auc
from numpy import array


def check_auc(true_auc, pred_bbox, true_bbox):
    calculated_auc = calc_auc(pred_bbox, true_bbox)
    diff = abs(calculated_auc - true_auc)
    if not diff <= 0.04:
        raise ValueError("Incorrect AUC, expected: "+ str(true_auc) + \
                         ", got: " + str(calculated_auc) + \
                         "\nAll bboxes:\n" + ''.join([str(line) + "\n" for line in pred_bbox["img1"]]) + \
                         "\nTrue bboxes:\n" + str(array(true_bbox["img1"])))


def test_auc1():
    pred_bbox = {"img1": array([[8, 112, 40, 100, 0.4083907],
                                [56, 108, 40, 100, 0.40634221],
                                [4, 20, 40, 100, 0.40198728],
                                [68, 24, 40, 100, 0.4007026],
                                [64, 160, 40, 100, 0.42576572],
                                [16, 12, 40, 100, 0.4111293],
                                [12, 16, 40, 100, 0.42142084],
                                [60, 36, 40, 100, 0.41478667]])}
    true_bbox = {"img1": [[69, 68, 40, 100],
                          [62, 31, 40, 100],
                          [61, 137, 40, 100]]}
    check_auc(0.528, pred_bbox, true_bbox)


def test_auc2():
    pred_bbox = {"img1": array([[60, 8, 40, 100, 0.42845643],
                                [44, 96, 40, 100, 0.41994753],
                                [12, 76, 40, 100, 0.40966052],
                                [36, 104, 40, 100, 0.4556621],
                                [80, 28, 40, 100, 0.41632652],
                                [56, 60, 40, 100, 0.41132379],
                                [60, 168, 40, 100, 0.4010765],
                                [24, 112, 40, 100, 0.42834586],
                                [100, 176, 40, 100, 0.4156765],
                                [100, 60, 40, 100, 0.4023141]])}
    true_bbox = {"img1": [[62, 30, 40, 100],
                          [67, 177, 40, 100],
                          [106, 20, 40, 100],
                          [111, 259, 40, 100]]}
    check_auc(0.101, pred_bbox, true_bbox)


def test_auc3():
    pred_bbox = {"img1": array([[52, 4, 40, 100, 0.43235692],
                                [104, 72, 40, 100, 0.41292477],
                                [52, 152, 40, 100, 0.40181845]])}
    true_bbox = {"img1": [[58, 7, 40, 100],
                          [58, 105, 40, 100],
                          [59, 198, 40, 100]]}
    check_auc(0.333, pred_bbox, true_bbox)


def test_auc4():
    pred_bbox = {"img1": array([[58, 7, 40, 100, 0.43235692],
                                [58, 105, 40, 100, 0.41292477],
                                [59, 198, 40, 100, 0.40181845]])}
    true_bbox = {"img1": [[58, 7, 40, 100],
                          [58, 105, 40, 100],
                          [59, 198, 40, 100]]}
    check_auc(1.0, pred_bbox, true_bbox)


def test_auc5():
    pred_bbox = {"img1": array([[52, 4, 40, 100, 0.39235692],
                                [104, 72, 40, 100, 0.41292477],
                                [52, 152, 40, 100, 0.40181845]])}
    true_bbox = {"img1": [[58, 7, 40, 100],
                          [58, 105, 40, 100],
                          [59, 198, 40, 100]]}
    check_auc(0.056, pred_bbox, true_bbox)


def test_auc6():
    pred_bbox = {"img1": array([[25, 72, 40, 60, 0.2],
                                [75, 145, 60, 80, 0.3],
                                [79, 149, 50, 80, 0.4],
                                [62, 42, 70, 70, 0.1]])}
    true_bbox = {"img1": [[20, 70, 50, 80],
                          [40, 100, 50, 80],
                          [60, 50, 50, 80],
                          [80, 150, 50, 80],
                          [0, 200, 50, 80]]}
    check_auc(0.458, pred_bbox, true_bbox)


def test_auc7():
    pred_bbox = {"img1": array([[11, 15, 40, 100, 0.5],
                                [12, 15, 40, 100, 0.5],
                                [13, 15, 40, 100, 0.5],
                                [14, 15, 40, 100, 0.5],
                                [16, 15, 40, 100, 0.5],
                                [17, 15, 40, 100, 0.5],
                                [18, 15, 40, 100, 0.5],
                                [19, 15, 40, 100, 0.5]])}
    true_bbox = {"img1": [[15, 15, 40, 100]]}
    check_auc(0.5625, pred_bbox, true_bbox)


def test_auc8():
    x, y = np.mgrid[:100, :100]
    x = x.flatten()
    y = y.flatten()
    h = w = 10 * np.ones_like(x)
    conf = np.abs(x - 45) + 100 * np.abs(y - 50)
    conf = 1 - conf / conf.max()
    pred = np.stack([x, y, h, w, conf], axis=-1)

    pred_bbox = {"img1": pred}
    true_bbox = {"img1": [[50, 50, 10, 10]]}

    start = time.perf_counter()
    check_auc(0.1, pred_bbox, true_bbox)
    end = time.perf_counter()

    if end - start > 1.0:
        raise ValueError(
            "Please check, that your implementation doesn't have quadratic "
            "complexity with respect to the number of predictions."
        )


def test_auc9():
    pred_bbox = {"img1": array([[0, 0, 40, 100, 0.44231782],
                                [1, 0, 40, 100, 0.42432535],
                                [58, 7, 40, 100, 0.43235692]]),
                 "img2": array([[0, 0, 40, 100, 0.54789672],
                                [1, 0, 40, 100, 0.41124892],
                                [58, 105, 40, 100, 0.41292477]]),
                 "img3": array([[100, 7, 40, 100, 0.99245356]]),
                 "img4": array([[0, 0, 40, 100, 0.30412384],
                                [59, 198, 40, 100, 0.40181845]])}
    true_bbox = {"img1": [[58, 7, 40, 100]],
                 "img2": [[58, 105, 40, 100]],
                 "img3": [[100, 7, 40, 100], [200, 7, 40, 100]],
                 "img4": [[59, 198, 40, 100]]}
    check_auc(0.46619047619047616, pred_bbox, true_bbox)
