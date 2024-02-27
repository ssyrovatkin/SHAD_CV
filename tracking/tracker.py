import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""

    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        detections = extract_detections(frame)
        for i in range(len(detections)):
            detections[i][0] = self.new_label()

        return detections

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = []
        # Write code here
        tail_detections = self.detection_history[-self.lookup_tail_size:]
        tail_detections = list(reversed(tail_detections))

        for curr_id in range(self.tracklet_count):
            for detection in tail_detections:
                for id in detection[:, 0]:
                    if curr_id == id:
                        detections.append(detection)
                        break
                break

        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here

        # Step 1: calc pairwise detection IOU
        iou_list = []

        for prev_detect in prev_detections:
            prev_detect_id = prev_detect[0]
            for detect_idx, detect in enumerate(detections):
                iou_value = iou_score(prev_detect[1:], detect[1:])
                iou_list.append([iou_value, detect_idx, prev_detect_id])

        # Step 2: sort IOU list
        iou_list = sorted(iou_list, key=lambda x: x[0], reverse=True)

        # Step 3: fill detections[:, 0] with best match
        # One matching for each id
        for i in range(len(detections)):
            detections[i][0] = -1

        detected_prev_id = []
        for iou_value, detect_idx, prev_detect_id in iou_list:
            if prev_detect_id not in detected_prev_id:
                detections[detect_idx, 0] = prev_detect_id
                detected_prev_id.append(prev_detect_id)

        # Step 4: assign new tracklet id to unmatched detections
        for i in range(len(detections)):
            if detections[i][0] == -1:
                detections[i][0] = self.new_label()

        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
