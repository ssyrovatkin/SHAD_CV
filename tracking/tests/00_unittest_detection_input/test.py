from detection import extract_detections
import numpy as np
from os.path import dirname, join
from skimage.io import imread


def test_detection():
    data_dir = dirname(__file__)
    detections = extract_detections(imread(join(data_dir, 'new_york.jpg')))

    gt_detections = np.array([
        [15, 118, 208, 158, 289],
        [15, 55, 178, 89, 289],
        [15, 165, 187, 200, 282],
        [7, 192, 180, 255, 226]
    ])

    threshold = 10

    assert detections.shape == gt_detections.shape

    # Invariant to the order of detections
    for gt_det in gt_detections:
        difference = np.abs(gt_det[1:] - detections[:, 1:])
        assert np.sum(np.all(difference < threshold, axis=-1)) == 1
