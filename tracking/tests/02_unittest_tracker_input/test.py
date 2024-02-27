from glob import glob
import numpy as np
from os.path import dirname, join
from skimage.io import imread
from tracker import Tracker


def test_tracker():
    data_dir = dirname(__file__)

    frames = []
    frame_paths = sorted(glob(join(data_dir, 'frames', '*.jpg')))
    for frame_path in frame_paths:
        frames.append(imread(frame_path))

    tracker = Tracker(return_images=False)
    result_tracks = [tracker.update_frame(frame) for frame in frames]

    gt_tracks = [
        [[0, 1061, 400, 1269, 631], [1, 853, 416, 942, 470]],
        [[0, 1046, 402, 1280, 633], [1, 847, 416, 938, 468]],
        [[0, 1026, 398, 1279, 645], [1, 847, 416, 938, 468]],
        [[0, 1020, 404, 1280, 643], [1, 847, 415, 937, 467]],
        [[0, 1009, 402, 1280, 627], [1, 848, 410, 942, 464]]
    ]
    gt_tracks = [np.array(detections) for detections in gt_tracks]

    threshold = 25
    label_map = {}

    # Invariant to the order of detections and label values
    for gt_detections, result_detections in zip(gt_tracks, result_tracks):
        assert gt_detections.shape == result_detections.shape

        for gt_det in gt_detections:
            difference = np.abs(gt_det[1:] - result_detections[:, 1:])
            equal = np.all(difference < threshold, axis=-1)

            assert np.sum(equal) == 1

            index = np.argmax(equal)
            res_label = result_detections[index, 0]

            assert label_map.setdefault(gt_det[0], res_label) == res_label

    labels = label_map.values()
    assert len(labels) == len(set(labels))
