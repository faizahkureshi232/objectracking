from __future__ import division, print_function, absolute_import
import numpy as np

from deep_sort.application_util import preprocessing
from deep_sort.application_util.visualization import draw_trackers
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

def gather_sequence_info(detections, image):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).
    """

    image_size = image.shape[:2]
    min_frame_idx = 1
    max_frame_idx = 1
    update_ms = 5
    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": "NA",
        "image_filenames": "NA",
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def create_detections(detection_mat, min_height=0, frame_idx=1):
    """Create detections for given frame index from the raw detection matrix.
    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(image, detection, config, min_confidence,
        nms_max_overlap, min_detection_height):

    img_cpy = image.copy()
    seq_info = gather_sequence_info(detection, img_cpy)
    tracker = config.tracker

    # Run tracker.
    # Load image and generate detections.
    detections = create_detections(
        seq_info["detections"], min_detection_height)
    detections = [d for d in detections if d.confidence >= min_confidence]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    # Check if non_max_suppression is needed again
    indices = preprocessing.non_max_suppression(
        boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Update tracker.
    tracker.predict()
    tracker.update(detections)

    draw_trackers(tracker.tracks, img_cpy)

def run_deep_sort(image, detection, config):
    min_confidence = 0.1
    nms_max_overlap = 1.0
    min_detection_height = 0.0
    run(image, detection, config, min_confidence, nms_max_overlap, min_detection_height)

class DeepSORTConfig:
    def __init__(self, max_cosine_distance=0.2, nn_budget = 100):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.results = []
