from os.path import dirname, join
import numpy as np

from moviepy.editor import VideoFileClip
from tqdm import tqdm

from cross_correlation import CorrelationTracker
from metrics import motp_mota
from utils import load_result


def test_tracking():
    data_dir = join(dirname(__file__), 'data')
    video_names = ['road', 'jogging', 'street']

    gt_data = np.array([
        [0.936, 0.968],
        [0.676, 0.697],
        [0.694, 0.328]
    ])

    threshold = 0.05

    for video_name, gt_metrics in zip(video_names, gt_data):
        input_clip = VideoFileClip(join(data_dir, video_name + '.mp4'))

        tracker = CorrelationTracker(return_images=False)
        result_tracks = [tracker.update_frame(frame) for frame in tqdm(input_clip.iter_frames())]
        result_tracks = [detections.tolist() for detections in result_tracks]

        gt_tracks = load_result(join(data_dir, video_name + '.csv'))
        gt_tracks = [detections.tolist() for detections in gt_tracks]

        result_metrics = np.array(motp_mota(gt_tracks, result_tracks))

        assert (result_metrics > gt_metrics - threshold).all()
