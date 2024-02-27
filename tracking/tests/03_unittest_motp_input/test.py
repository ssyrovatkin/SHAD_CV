from metrics import motp
from numpy.testing import assert_allclose


def test_motp():
    gt_tracks = [
        [[0, 0, 0, 150, 100], [1, 400, 0, 500, 150]],
        [[0, 50, 50, 200, 150], [1, 400, 100, 500, 250]],
        [[0, 150, 100, 300, 200], [1, 350, 150, 450, 300]]
    ]
    result_tracks = [
        [[0, 15, 15, 145, 115], [1, 380, 20, 510, 150]],
        [[0, 40, 40, 190, 160], [1, 400, 140, 520, 270]],
        [[0, 150, 150, 280, 240], [1, 330, 170, 450, 290]]
    ]

    gt_metric = 0.665
    result_metric = motp(gt_tracks, result_tracks)

    assert_allclose(result_metric, gt_metric, rtol=0, atol=0.01)
