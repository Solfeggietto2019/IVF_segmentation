import cv2
import numpy as np
from typing import List, Tuple, Any, Dict
import math
from utils.dataclasses import Sperm, SelectedSperm
from utils.dataclasses import Egg, VersionControl, DataStructure


def adjust_coordinates(
    x: int, y: int, frame_shape: tuple, dim: int = 10, scale_width: int = 640
) -> tuple:
    """
    Adjust the coordinates (x, y) based on the frame dimensions and scaling factor.

    Parameters:
    - x (int): The original x-coordinate.
    - y (int): The original y-coordinate.
    - frame_shape (tuple): The shape of the frame as (height, width, channels).
    - dim (int, optional): The dimension to adjust the coordinates by. Default is 10.
    - scale_width (int, optional): The width to scale the coordinates by. Default is 640.

    Returns:
    - tuple: Adjusted (x, y) coordinates.
    """
    frame_height, frame_width, _ = frame_shape
    scaling_factor = frame_width / scale_width

    x = round(x * scaling_factor)
    y = round(y * scaling_factor)

    if x + dim > frame_width:
        x = frame_width - dim
    if x - dim < 0:
        x = dim
    if y + dim > frame_height:
        y = frame_height - dim
    if y - dim < 0:
        y = dim

    return x, y


def draw_positions(
    frame: np.ndarray, positions: Tuple[int, int], sperm_id: int
) -> np.ndarray:
    """
    Draw a red circle and the sperm ID on the frame at the given position.

    :param frame: The current video frame.
    :param positions: A tuple (x, y) where the circle should be drawn.
    :param sperm_id: The ID of the sperm to be annotated.
    :return: The frame with the red circle and ID drawn.
    """
    x, y = positions

    # Draw a red circle at the position
    cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    # Put the sperm ID text near the circle
    cv2.putText(
        frame,
        f"ID: {sperm_id}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )
    return frame


def find_bbox_center(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> Tuple[float, float]:
    """
    Calculate the center coordinates of a bounding box.

    Args:
        x_min (float): The minimum x-coordinate (left) of the bounding box.
        y_min (float): The minimum y-coordinate (top) of the bounding box.
        x_max (float): The maximum x-coordinate (right) of the bounding box.
        y_max (float): The maximum y-coordinate (bottom) of the bounding box.

    Returns:
        Tuple[float, float]: A tuple containing the x and y coordinates of the center of the bounding box.
    """
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center


def calculate_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """
    Calculate the Euclidean distance between two points in a 2D space.

    Args:
        point1 (Tuple[float, float]): The coordinates of the first point (x1, y1).
        point2 (Tuple[float, float]): The coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def find_nearest_object(
    sperm_bbox_center_point: Tuple[float, float],
    sperms_data: Dict[str, List[Sperm]],
    current_frame: int,
) -> Tuple[Sperm, int]:
    min_distance = float("inf")
    nearest_object = None
    initial_frame = None

    for sperm_list in sperms_data.values():
        for sperm in sperm_list:
            if sperm.initial_frame <= (current_frame - 1) <= sperm.final_frame:
                position_idx = (current_frame - 1) - sperm.initial_frame
                if 0 <= position_idx < len(sperm.positions):
                    frame_number, x, y = sperm.positions[position_idx]
                    position = (x, y)
                    distance = calculate_distance(sperm_bbox_center_point, position)
                    if distance < min_distance and distance < 10:
                        min_distance = distance
                        nearest_object = sperm
                        initial_frame = sperm.initial_frame

    return nearest_object, initial_frame


def get_morphological_features_from_mask(sperm: SelectedSperm) -> None:
    """
    Process a video frame to extract and compute various geometric properties of the detected contours.

    Args:
        mask (np.ndarray): The mask image containing regions of interest.
        x1 (int): The minimum x-coordinate of the region.
        y1 (int): The minimum y-coordinate of the region.
        x2 (int): The maximum x-coordinate of the region.
        y2 (int): The maximum y-coordinate of the region.
        box

    Returns:
        None
    """
    padding = 10
    mask = sperm.mask
    x_min, y_min, x_max, y_max = sperm.bbox
    mask_height, mask_width = mask.shape
    x1 = max(int(x_min * mask_width) - padding + 4, 0)
    y1 = max(int(y_min * mask_height) - padding - 2, 0)
    x2 = min(int(x_max * mask_width) + padding - 4, mask_width)
    y2 = min(int(y_max * mask_height) + padding + 4, mask_height)

    aspect_ratio = float(x_max - x_min) / (y_max - y_min)
    region = mask[y1:y2, x1:x2]
    mask_info = {}

    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest_contour)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        extent = float(area) / bbox_area
        perimeter = cv2.arcLength(largest_contour, True)

        if perimeter != 0:
            circularity = 4 * math.pi * area / (perimeter**2)
        else:
            circularity = 0

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)

        try:
            solidity = float(area) / hull_area
            convexity = perimeter / hull_perimeter
        except ZeroDivisionError:
            solidity = 0
            convexity = 0

        orientation_angle = 0
        if isinstance(largest_contour, cv2.UMat):
            largest_contour = largest_contour.get()

        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (_, axes, orientation) = ellipse
            orientation_angle = orientation
            major_axis_length, minor_axis_length = max(axes), min(axes)

            try:
                eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
                major_axis_radius = major_axis_length / 2.0
                minor_axis_radius = minor_axis_length / 2.0
            except ZeroDivisionError:
                eccentricity = 0
                major_axis_radius = 0
                minor_axis_radius = 0
        else:
            eccentricity = 0
            major_axis_radius = 0
            minor_axis_radius = 0
        compactness = np.sqrt(4 * area / np.pi) / perimeter if perimeter != 0 else 0

        mask_info = {
            "area": area,
            "perimeter": perimeter,
            "aspect_ratio": aspect_ratio,
            "extend": extent,
            "orientated_angle": orientation_angle,
            "circularity": circularity,
            "hull_area": hull_area,
            "solidity": solidity,
            "hull_perimeter": hull_perimeter,
            "convexity": convexity,
            "eccentricity": eccentricity,
            "compactness": compactness,
            "major_axis_radius": major_axis_radius,
            "minor_axis_radius": minor_axis_radius,
        }
    return mask_info

def make_response_json(sperm_info, egg_info, frame_number):
    version_control = VersionControl()
    egg_mask = egg_info['oocytes'][0]['masks']
    egg_features = egg_info['oocytes'][0]['features']

    egg = Egg(frame_number, egg_mask, egg_features)
    data_structure = DataStructure(
        VersionControl=version_control,
        SiD=sperm_info,
        Aeris=egg
    )
    json_output = data_structure.to_json()
    return json_output 

def make_final_json(sperms, eggs, sofi_responses):
    version_control = VersionControl()
    data_structure = DataStructure(
        objectID = "0Wxootb0CP",
        VersionControl=version_control,
        SiD=sperms,
        Aeris=eggs,
        Sofi=sofi_responses
    )
    json_output = data_structure.to_json()
    json_output_dict = data_structure.to_dict()
    return json_output, json_output_dict

def get_manual_selected_sperms(sperms_data, ids_list):
    selected_sperms = []
    for selected_id in ids_list:
        for sperm_list in sperms_data.values():
            for sperm in sperm_list:
                if sperm.id == selected_id:
                    selected_sperms.append(sperm)

    return selected_sperms