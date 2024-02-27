from numpy.testing import assert_allclose

from metrics import motp_mota


def test_motp_mota_1():
    gt_tracks = [
        [[0, 0, 0, 150, 100], [1, 400, 0, 500, 150]],
        [[0, 50, 50, 200, 150], [1, 400, 100, 500, 250]],
        [[0, 150, 100, 300, 200], [1, 350, 150, 450, 300]]
    ]
    result_tracks = [
        [[0, 15, 15, 145, 115], [1, 380, 20, 510, 150]],
        [[0, 40, 40, 190, 160], [1, 400, 140, 520, 270]],
        [[0, 150, 150, 280, 240], [2, 330, 170, 450, 290]]
    ]

    gt_metrics = [0.665, 0.5]
    result_metrics = motp_mota(gt_tracks, result_tracks)

    assert_allclose(result_metrics, gt_metrics, rtol=0, atol=0.01)


def test_motp_mota_2():
    gt_tracks = [
        [[2, 832, 411, 997, 508]],
        [[2, 830, 411, 995, 508], [3, 938, 403, 1060, 477]]
    ]
    result_tracks = [
        [[1, 832, 410, 1001, 510]],
        [[1, 831, 410, 988, 510], [1, 938, 402, 1060, 478]]
    ]

    gt_metrics = [0.960, 2/3]
    result_metrics = motp_mota(gt_tracks, result_tracks)

    assert_allclose(result_metrics, gt_metrics, rtol=0, atol=0.01)


def test_motp_mota_3():
    gt_tracks = [
        [[2, 832, 411, 997, 508]],
        [[2, 832, 411, 997, 508]],
        [[2, 830, 411, 995, 508]],
        [],
        [],
        [[2, 824, 411, 989, 508]],
        [[2, 822, 411, 987, 508]],
    ]
    result_tracks = [
        [[1, 832, 410, 1001, 510]],
        [[1, 832, 410, 1001, 510]],
        [[1, 831, 410,  988, 510]],
        [[1, 829, 410,  990, 510]], # fp
        [[1, 829, 410,  990, 510]], # fp
        [[1, 829, 410,  990, 510], [3, 824, 411,  989, 508]],
        [[1, 822, 410,  987, 510]],
    ]

    gt_metrics = [0.945, 0.4]
    result_metrics = motp_mota(gt_tracks, result_tracks)

    assert_allclose(result_metrics, gt_metrics, rtol=0, atol=0.01)
