import cv2
import numpy as np
from typing import List, Tuple, Any
from utils.utils import (
    find_bbox_center,
    find_nearest_object,
    get_morphological_features_from_mask,
    make_response_json,
    calculate_distance
)
from utils.dataclasses import SelectedSperm
import time
from api_calls.aeris import api_maturity, api_key, convert_image_to_base64
from utils.standardize_metrics import standardize_sperm_metrics

egg_class = 4
pipette_class = 5
pipette_detected_frame = -1
cooldown_frames = 30
fps = 30
save_frame_delay = 3 * fps
last_collision_frame = -cooldown_frames
frame_saved = False
            

def process_inference_results_manually(
    selected_sperm: SelectedSperm,
    manual_selected_sperms,
    current_frame,
    results: Any,
    width: int,
    height: int,
    original_frame: np.ndarray,
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
    global last_collision_frame, pipette_detected_frame, frame_saved

    if results[0].boxes is not None and results[0].masks is not None:
        boxes = results[0].boxes.xyxyn.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        # track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy().astype(float)
        masks = (results[0].masks.data.cpu().numpy() * 255).astype("uint8")
    else:
        boxes, track_ids, confidences, masks, cls = [], [], [], [], []

    annotated_frame = results[0].plot(boxes=False)


    for box, mask, class_id, confidence in zip(boxes, masks, cls, confidences):
        x_min, y_min, x_max, y_max = box
        text_position = (int(x_min * width), int(y_min * height))
        mask_height, mask_width = mask.shape
        if class_id == 0:
            sperm_bbox = (
                int(x_min * width),
                int(y_min * height),
                int(x_max * width),
                int(y_max * height),
            )
            for sperm in manual_selected_sperms:
                if current_frame == sperm.initial_frame:
                    #position_idx = (current_frame - 1) - sperm.initial_frame
                    if 0 <= current_frame < len(sperm.positions):
                        frame_number, x, y = sperm.positions[current_frame]
                        x_min, y_min, x_max, y_max = sperm_bbox
                        sperm_bbox_center_point = find_bbox_center(
                                x_min, y_min, x_max, y_max
                            )
                        position = (x, y)
                        distance = calculate_distance(sperm_bbox_center_point, position)
                        if distance < 10:
                            selected_sperm.Id = sperm.id
                            selected_sperm.motility_parameters = (
                                sperm.standard_motility_parameters
                            )
                            filename_sperm = f"sperm_image.png"
                            cv2.imwrite(filename_sperm, original_frame)
                            sperm_b64_frame_image = convert_image_to_base64(filename_sperm)
                            selected_sperm.mask = mask
                            selected_sperm.bbox = box
                            selected_sperm.frame = current_frame
                            selected_sperm.sid_score = sperm.SiDScore
                            selected_sperm.initial_frame = sperm.initial_frame
                            selected_sperm.b64_string_frame = sperm_b64_frame_image
                            pipette_detected_frame = -1
        if class_id == 4:
            if selected_sperm and selected_sperm.Id is not None:
                print(f"Track ID of the selected sperm: {selected_sperm.Id}")
                sperm_morph_info = get_morphological_features_from_mask(selected_sperm)
                standardize_morph_sperm_metrics = standardize_sperm_metrics(
                    sperm_morph_info
                )
                cv2.putText(
                    annotated_frame,
                    f"Selected Sperm Track ID: {selected_sperm.Id}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                selected_sperm.morphological_parameters = sperm_morph_info
                selected_sperm.standardize_morph_parameters = (
                    standardize_morph_sperm_metrics
                )
                if pipette_class in cls:
                    if pipette_detected_frame == -1:
                        pipette_detected_frame = current_frame
                        frame_saved = False
                    else:
                        if (
                            current_frame - pipette_detected_frame > save_frame_delay
                            and not frame_saved
                        ):
                            timestamp = int(time.time())
                            filename_egg = f"inyected_egg_{timestamp}.png"
                            cv2.imwrite(filename_egg, original_frame)
                            egg_b64_frame_image = convert_image_to_base64(filename_egg)
                            response = api_maturity(filename_egg, api_key)
                            if response:
                                print("Resultado del análisis:", response)

                            print(f"Frame guardado como {filename_egg}")
                            frame_saved = True
                            selected_sperm = selected_sperm.to_serializable()
                            return annotated_frame, selected_sperm, response, current_frame, egg_b64_frame_image

    return annotated_frame, None, None, None, None