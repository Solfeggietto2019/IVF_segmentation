import cv2
from data_processing.json_reader import JSONReader
from data_processing.stream_reader import VideoStreamReader
from data_processing.yolo_model import YOLOModel
from data_processing.inference_logic import process_inference_results
from data_processing.manual_processing import process_inference_results_manually
from utils.utils import (
    draw_positions,
    adjust_coordinates,
    make_final_json,
    get_manual_selected_sperms,
)
from utils.dataclasses import SelectedSperm, Egg
from api_calls.sofi import call_sofi
import json
import argparse
import os


MODEL_PATH = "models/best.pt"


def process(json_file_path: str, video_file_path: str) -> str:
    # json_file_path = "data/json/IsaacVideoTest3_2024-06-25-22-56-Camera.json"
    # video_file_path = "data/video/IsaacVideoTest3_2024-06-25-22-56-Camera.mp4"
    # model_path = "models/best.pt"

    selected_sperm = SelectedSperm()
    reader = JSONReader(json_file_path)
    reader.download_video(video_file_path)
    video_reader = VideoStreamReader(video_file_path)
    yolo = YOLOModel(MODEL_PATH)

    parser = argparse.ArgumentParser(description="Process some sperm data.")
    parser.add_argument(
        "--no-auto", action="store_true", help="Disable automatic processing."
    )
    parser.add_argument(
        "--ids",
        type=str,
        help="List of ID's separated by commas in injection order (e.g.: 1,2,3).",
    )

    args = parser.parse_args()

    selected_sperms = []
    egg_responses = []
    frame_numbers = []
    egg_b64_frames = []

    sperms_data, object_id = reader.extract_sperms_data()

    if args.no_auto and args.ids:
        ids_list = [id.strip() for id in args.ids.split(",")]
        print(f"Selected Id's: {ids_list}")
        manual_selected_sperms = get_manual_selected_sperms(sperms_data, ids_list)
    try:
        for video_data in video_reader:
            frame, width, height, fps, num_frame = video_data
            original_frame = frame.copy()
            num_frame -= 1
            results = yolo.infer(frame)

            for sperm_list in sperms_data.values():
                for sperm in sperm_list:
                    if sperm.initial_frame <= num_frame <= sperm.final_frame:
                        position_idx = num_frame - sperm.initial_frame
                        if 0 <= position_idx < len(sperm.positions):
                            frame_number, x, y = sperm.positions[position_idx]
                            x_adjusted, y_adjusted = adjust_coordinates(
                                x, y, frame.shape
                            )
                            sperm.positions[position_idx] = (
                                frame_number,
                                x_adjusted,
                                y_adjusted,
                            )
                            frame = draw_positions(
                                frame, (x_adjusted, y_adjusted), sperm.id
                            )

            if args.no_auto and args.ids:
                annotated_frame, sperm_object, egg_object, selected_frame, egg_b64 = (
                    process_inference_results_manually(
                        selected_sperm,
                        manual_selected_sperms,
                        num_frame,
                        results,
                        width,
                        height,
                        original_frame,
                    )
                )

            else:
                annotated_frame, sperm_object, egg_object, selected_frame, egg_b64 = (
                    process_inference_results(
                        selected_sperm,
                        sperms_data,
                        num_frame,
                        results,
                        frame,
                        width,
                        height,
                        original_frame,
                    )
                )
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if sperm_object and egg_object and selected_frame and egg_b64:
                selected_sperms.append(sperm_object)
                egg_responses.append(egg_object)
                frame_numbers.append(selected_frame)
                egg_b64_frames.append(egg_b64)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_reader.release()
        cv2.destroyAllWindows()

    eggs = [
        (
            Egg(
                frame_number,
                response["oocytes"][0]["masks"],
                response["oocytes"][0]["features"],
                egg_b64,
            )
            if response["oocytes"]
            else None
        )
        for response, frame_number, egg_b64 in zip(
            egg_responses, frame_numbers, egg_b64_frames
        )
    ]
    sperms = [sperm_object for sperm_object in selected_sperms]
    sofi_responses = call_sofi(sperms, eggs)
    sofi_responses_content = sofi_responses_content = [response.json() for response in sofi_responses]
    json_output, json_dict = make_final_json(sperms, eggs, sofi_responses_content, object_id)
    filename = "test_final-56-manual.json"
    json_string = json.dumps(json_dict, indent=4)
    with open(filename, "w") as file:
        file.write(json_string)
    # Asegurar el cierre del archivo JSON
    ensure_file_closed(json_file_path)

    # Retornar el contenido de la respuesta
    return json_string


def ensure_file_closed(file_path: str):
    try:
        with open(file_path, "r") as f:
            pass
    except Exception as e:
        print(f"Error al asegurar el cierre del archivo: {e}")


def clean_up_files(json_file_path: str, video_file_path: str):
    # Asegurar el cierre del archivo JSON
    ensure_file_closed(json_file_path)

    # Eliminar archivos temporales
    if os.path.exists(json_file_path):
        try:
            os.remove(json_file_path)
        except Exception as e:
            print(f"Error al eliminar el archivo JSON: {e}")
    if os.path.exists(video_file_path):
        try:
            os.remove(video_file_path)
        except Exception as e:
            print(f"Error al eliminar el archivo de video: {e}")
