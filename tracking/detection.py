import os

import cv2
import numpy as np
import torch
from config import VOC_CLASSES, bbox_util, model
from PIL import Image
from skimage import io
from skimage.transform import resize
from utils import get_color


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)


def image2tensor(image):
    # Write code here
    image = image.astype('float64')  # convert frame to float
    image = resize(image, (300, 300))  # resize image to 300x300
    image = image[..., ::-1]   # convert RGB to BGR
    image = image - IMAGENET_MEAN  # center with respect to imagenet means
    image = image.transpose([2, 0, 1])  # torch works with CxHxW images
    tensor = torch.FloatTensor(image.copy()).unsqueeze(0)
    # tensor.shape == (1, channels, height, width)
    return tensor


@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    # Write code here
    # First, convert the input image to tensor
    W, H, _ = frame.shape
    input_tensor = image2tensor(frame)

    # Then use model(input_tensor),
    # convert output to numpy
    # and bbox_util.detection_out

    output = model(input_tensor).detach().numpy()
    results = bbox_util.detection_out(output, confidence_threshold=min_confidence)

    # Select detections with confidence > min_confidence
    # hint: see confidence_threshold argument of bbox_util.detection_out

    # If label set is known, use it
    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indices = [
            index
            for index, label in enumerate(result_labels)
            if VOC_CLASSES[label - 1] in labels
        ]
        results = results[indices]

    # Remove confidence column from result
    for i in range(len(results)):
        if len(results[i]) != 0:
            results[i] = np.delete(results[i], 1, 1)
            results[i][:, 1] = np.round(H * results[i][:, 1])
            results[i][:, 2] = np.round(W * results[i][:, 2])
            results[i][:, 3] = np.round(H * results[i][:, 3])
            results[i][:, 4] = np.round(W * results[i][:, 4])
    # Resize detection coords to the original image shape.

    # Return result
    return detection_cast(results)


def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """
    frame = frame.copy()

    # Write code here

    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, "data", "test.png"))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == "__main__":
    main()
