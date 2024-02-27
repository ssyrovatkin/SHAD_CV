from moviepy.editor import VideoFileClip
from os.path import dirname, join
from tqdm import tqdm

from tracker import Tracker
from metrics import motp
from utils import load_result


def test_baseline():
    data_dir = dirname(__file__)
    video_name = 'road'

    input_clip = VideoFileClip(join(data_dir, video_name + '.mp4'))

    tracker = Tracker(return_images=False)
    result_tracks = [tracker.update_frame(frame) for frame in tqdm(input_clip.iter_frames())]
    result_tracks = [detections.tolist() for detections in result_tracks]

    gt_tracks = load_result(join(data_dir, video_name + '.csv'))
    gt_tracks = [detections.tolist() for detections in gt_tracks]

    result_motp = motp(gt_tracks, result_tracks)

    gt_motp = 0.937
    threshold = 0.05

    assert result_motp > gt_motp - threshold
