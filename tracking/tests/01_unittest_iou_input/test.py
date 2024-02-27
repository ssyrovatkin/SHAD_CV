from numpy.testing import assert_allclose

from metrics import iou_score


def test_iou():
    base_bbox = [16, 25, 42, 80]
    bboxes = [
        [23, 36, 40, 71],
        [2, 8, 12, 100],
        [10, 27, 44, 82],
        [14, 6, 78, 42],
        [18, 22, 44, 80]
    ]

    gt_metrics = [0.416, 0, 0.717, 0.134, 0.816]
    result_metrics = [iou_score(base_bbox, bbox) for bbox in bboxes]

    assert_allclose(result_metrics, gt_metrics, rtol=0, atol=0.01)
