from numpy import array
from detection_and_metrics import nms


def check_nms(predictions, true_nms):
    got_after_nms = nms({"1": predictions}, 0.6)["1"]
    got_after_nms = array(got_after_nms).astype("int")[:, :4].tolist()
    extra_bboxes = got_after_nms.copy()
    for bbox in true_nms:
        if bbox not in extra_bboxes:
            raise ValueError("Incorrect bboxes after NMS: " + \
                             "bbox " + str(bbox) + " should exist after NMS." + \
                             "\nAll bboxes before NMS:\n" + ''.join([str(line) + "\n" for line in predictions]) + \
                             "True bboxes after NMS:\n" + str(array(true_nms)))
        del extra_bboxes[extra_bboxes.index(bbox)]
    if len(extra_bboxes) > 0:
            raise ValueError("Got extra bboxes after NMS: " + str(array(extra_bboxes)) + \
                             "\nAll bboxes before NMS:\n" + str(array(predictions)) + \
                             "\nTrue bboxes after NMS:\n" + str(array(true_nms)))


def test_test1():
    predictions = [[0.0, 0.0, 40.0, 100.0, 0.0001439873068488249],
                   [0.0, 0.0, 40.0, 100.0, 0.0001439873068488249],
                   [0.0, 8.0, 40.0, 100.0, 8.844506876789047e-05],
                   [0.0, 12.0, 40.0, 100.0, 8.864786832494879e-05],
                   [0.0, 16.0, 40.0, 100.0, 5.646329540611465e-05],
                   [0.0, 20.0, 40.0, 100.0, 5.32966834220056e-05],
                   [0.0, 24.0, 40.0, 100.0, 3.776895335082916e-05],
                   [0.0, 12.0, 40.0, 100.0, 8.864786832494879e-05],
                   [0.0, 24.0, 40.0, 100.0, 3.776895335082916e-05],
                   [0.0, 16.0, 40.0, 100.0, 5.646329540611465e-05],
                   [0.0, 32.0, 40.0, 100.0, 2.122202754350387e-05],
                   [0.0, 20.0, 40.0, 100.0, 5.32966834220056e-05],
                   [0.0, 20.0, 40.0, 100.0, 5.32966834220056e-05],
                   [0.0, 16.0, 40.0, 100.0, 5.646329540611465e-05],
                   [0.0, 28.0, 40.0, 100.0, 2.6099790148424712e-05],
                   [0.0, 52.0, 40.0, 100.0, 1.1105166866133818e-05],
                   [0.0, 24.0, 40.0, 100.0, 3.776895335082916e-05],
                   [0.0, 0.0, 40.0, 100.0, 0.0001439873068488249],
                   [0.0, 68.0, 40.0, 100.0, 7.211932406036818e-06],
                   [0.0, 64.0, 40.0, 100.0, 6.360868819693369e-06]]
    true_nms = [[0, 0, 40, 100], [0, 28, 40, 100], [0, 68, 40, 100]]
    check_nms(predictions, true_nms)


def test_test2():
    predictions = [[0.0, 0.0, 40.0, 100.0, 0.09782119690227115],
                   [0.0, 0.0, 40.0, 100.0, 0.09782119690227115],
                   [0.0, 8.0, 40.0, 100.0, 0.06903079641461966],
                   [0.0, 12.0, 40.0, 100.0, 0.030923605346572323],
                   [0.0, 4.0, 40.0, 100.0, 0.05965406988277452],
                   [0.0, 16.0, 40.0, 100.0, 0.01137162766724347],
                   [0.0, 24.0, 40.0, 100.0, 0.0014073698970341105],
                   [0.0, 4.0, 40.0, 100.0, 0.05965406988277452],
                   [0.0, 0.0, 40.0, 100.0, 0.09782119690227115],
                   [0.0, 4.0, 40.0, 100.0, 0.05965406988277452],
                   [0.0, 20.0, 40.0, 100.0, 0.007435719252642719],
                   [0.0, 24.0, 40.0, 100.0, 0.0014073698970341105],
                   [0.0, 16.0, 40.0, 100.0, 0.01137162766724347],
                   [0.0, 0.0, 40.0, 100.0, 0.09782119690227115],
                   [0.0, 4.0, 40.0, 100.0, 0.05965406988277452],
                   [0.0, 24.0, 40.0, 100.0, 0.0014073698970341105],
                   [0.0, 48.0, 40.0, 100.0, 1.6536604912525756e-05],
                   [0.0, 48.0, 40.0, 100.0, 1.6536604912525756e-05],
                   [0.0, 4.0, 40.0, 100.0, 0.05965406988277452],
                   [0.0, 40.0, 40.0, 100.0, 3.6870285221234696e-05]]
    true_nms = [[0, 0, 40, 100], [0, 40, 40, 100]]
    check_nms(predictions, true_nms)


def test_test3():
    predictions = [[0.0, 0.0, 40.0, 100.0, 0.006544825884670512],
                   [0.0, 4.0, 40.0, 100.0, 0.006980025392013287],
                   [0.0, 0.0, 40.0, 100.0, 0.006544825884670512],
                   [0.0, 0.0, 40.0, 100.0, 0.006544825884670512],
                   [0.0, 8.0, 40.0, 100.0, 0.006542695930173248],
                   [0.0, 20.0, 40.0, 100.0, 0.004849971539802249],
                   [0.0, 8.0, 40.0, 100.0, 0.006542695930173248],
                   [0.0, 12.0, 40.0, 100.0, 0.00633901766203902],
                   [0.0, 28.0, 40.0, 100.0, 0.0027040519169710133],
                   [0.0, 20.0, 40.0, 100.0, 0.004849971539802249],
                   [0.0, 36.0, 40.0, 100.0, 0.002317177146385066],
                   [0.0, 40.0, 40.0, 100.0, 0.002792573909938963],
                   [0.0, 8.0, 40.0, 100.0, 0.006542695930173248],
                   [0.0, 12.0, 40.0, 100.0, 0.00633901766203902],
                   [0.0, 28.0, 40.0, 100.0, 0.0027040519169710133],
                   [0.0, 20.0, 40.0, 100.0, 0.004849971539802249],
                   [0.0, 20.0, 40.0, 100.0, 0.004849971539802249],
                   [0.0, 48.0, 40.0, 100.0, 0.0012637383103649468],
                   [0.0, 52.0, 40.0, 100.0, 0.001776324209463092],
                   [0.0, 48.0, 40.0, 100.0, 0.0012637383103649468]]
    true_nms = [[0, 4, 40, 100], [0, 40, 40, 100]]
    check_nms(predictions, true_nms)