from glob import glob
from re import sub
import os
import pickle

import pytest
from seam_carve import seam_carve
from skimage.io import imread
import numpy as np


FILE_SUFFIX = "shrink_mask_hv_seams"
test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "public_data")


def get_seam_coords(seam_mask):
    coords = np.where(seam_mask)
    t = [i for i in zip(coords[0], coords[1])]
    t.sort(key=lambda i: i[0])
    return tuple(t)


def convert_img_to_mask(img):
    return ((img[:, :, 0] != 0) * -1 + (img[:, :, 1] != 0)).astype("int8")


def run_single_test(data_dir, orientation):
    img = imread(os.path.join(data_dir, "img.png"))
    mask = convert_img_to_mask(imread(os.path.join(data_dir, "mask.png")))
    seam = seam_carve(img, orientation + " shrink", mask=mask)[2]
    return get_seam_coords(seam)


def load_test_gt(gt_dir, orientation):
    with open(os.path.join(gt_dir, FILE_SUFFIX), "rb") as fgt:
        data = [
            pickle.load(fgt)
            for _ in range(2)
        ]
    return data[("horizontal", "vertical").index(orientation)]


@pytest.mark.parametrize("orientation", ("horizontal", "vertical"))
@pytest.mark.parametrize("test_num", list(range(1, 8)))
def test_real_shrink_masked(test_num, orientation):
    input_dir = os.path.join(data_dir, f"{test_num:02d}_test_img_input")
    gt_dir = sub("input$", "gt", input_dir)

    results = run_single_test(input_dir, orientation)
    expected = load_test_gt(gt_dir, orientation)

    assert results == expected
