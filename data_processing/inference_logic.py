import cv2
import numpy as np
from typing import List, Tuple, Any
from utils.utils import (
    find_bbox_center,
    find_nearest_object,
    get_morphological_features_from_mask,
)
from utils.dataclasses import SelectedSperm

egg_class = 4
cooldown_frames = 30
last_collision_frame = -cooldown_frames


def process_inference_results(
    selected_sperm: SelectedSperm,
    sperms_data,
    current_frame,
    results: Any,
    frame: np.ndarray,
    width: int,
    height: int,
    bbox_size: int = 30,
) -> np.ndarray:
    """
    Process the inference results and annotate the frame.

    :param results: The results from the YOLO model inference.
    :param frame: The current video frame.
    :param width: The width of the video frame.
    :param height: The height of the video frame.
    :return: The annotated frame.
    """
    global last_collision_frame
    if results[0].boxes is not None and results[0].masks is not None:
        boxes = results[0].boxes.xyxyn.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        # track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy().astype(float)
        masks = (results[0].masks.data.cpu().numpy() * 255).astype("uint8")
    else:
        boxes, track_ids, confidences, masks, cls = [], [], [], [], []

    annotated_frame = results[0].plot(boxes=False)

    needle_bboxes = []

    for box, mask, class_id, confidence in zip(boxes, masks, cls, confidences):
        x_min, y_min, x_max, y_max = box
        text_position = (int(x_min * width), int(y_min * height))
        mask_height, mask_width = mask.shape
        if class_id == 1:
            label = f"Needle, Conf: {confidence:.2f}"
            cv2.putText(
                annotated_frame,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

            needle_bbox = (
                int(x_min * width) - 5,
                int(y_min * height) - 5,
                int(x_min * width) + bbox_size,
                int(y_min * height) + bbox_size,
            )
            needle_bboxes.append(needle_bbox)
            cv2.rectangle(
                annotated_frame,
                (needle_bbox[0], needle_bbox[1]),
                (needle_bbox[2], needle_bbox[3]),
                (255, 255, 255),
                2,
            )

        elif class_id == 0:
            # label = f'Class: {str(class_id)}'
            # cv2.putText(annotated_frame, label, text_position,
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            sperm_bbox = (
                int(x_min * width),
                int(y_min * height),
                int(x_max * width),
                int(y_max * height),
            )

            for needle_bbox in needle_bboxes:
                if is_bbox_overlaping(needle_bbox, sperm_bbox) and egg_class not in cls:
                    if current_frame - last_collision_frame > cooldown_frames:
                        x_min, y_min, x_max, y_max = sperm_bbox
                        sperm_bbox_center_point = find_bbox_center(
                            x_min, y_min, x_max, y_max
                        )
                        overlaping_sperm = find_nearest_object(
                            sperm_bbox_center_point, sperms_data, current_frame
                        )
                        if overlaping_sperm:
                            selected_sperm.Id = overlaping_sperm.id
                            selected_sperm.Data = (
                                overlaping_sperm.standard_motility_parameters
                            )
                            selected_sperm.mask = mask
                            selected_sperm.bbox = box
                            selected_sperm.frame = current_frame
                        cv2.putText(
                            annotated_frame,
                            "Collision Detected",
                            (needle_bbox[0], needle_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (needle_bbox[0], needle_bbox[1]),
                            (needle_bbox[2], needle_bbox[3]),
                            (0, 0, 255),
                            2,
                        )
                        last_collision_frame = current_frame

        elif class_id == 4:
            if selected_sperm and selected_sperm.Id is not None:
                print(f"Track ID of the selected sperm: {selected_sperm.Id}")
                sperm_morph_info = get_morphological_features_from_mask(selected_sperm)
                cv2.putText(
                    annotated_frame,
                    f"Selected Sperm Track ID: {selected_sperm.Id}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                print(sperm_morph_info)

    return annotated_frame


def is_bbox_overlaping(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.

    :param bbox1: (x1_min, y1_min, x1_max, y1_max) coordinates of the first bounding box.
    :param bbox2: (x2_min, y2_min, x2_max, y2_max) coordinates of the second bounding box.
    :return: True if the bounding boxes overlap, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    if x1_max < x2_min or x1_min > x2_max:
        return False
    if y1_max < y2_min or y1_min > y2_max:
        return False
    return True
