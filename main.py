import cv2
from data_processing.json_reader import JSONReader
from data_processing.stream_reader import VideoStreamReader
from data_processing.yolo_model import YOLOModel
from data_processing.inference_logic import process_inference_results
from utils.utils import draw_positions, adjust_coordinates
from utils.dataclasses import SelectedSperm


def main() -> None:
    json_file_path = "data/json/IsaacVideoTest_2024-06-25-22-45-Camera.json"
    video_file_path = "data/video/IsaacVideoTest_2024-06-25-22-45-Camera.mp4"
    model_path = "models/best.pt"

    selected_sperm = SelectedSperm
    reader = JSONReader(json_file_path)
    video_reader = VideoStreamReader(video_file_path)
    yolo = YOLOModel(model_path)

    sperms_data = reader.extract_sperms_data()

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
                        x_adjusted, y_adjusted = adjust_coordinates(x, y, frame.shape)
                        sperm.positions[position_idx] = (
                            frame_number,
                            x_adjusted,
                            y_adjusted,
                        )
                        frame = draw_positions(
                            frame, (x_adjusted, y_adjusted), sperm.id
                        )

        annotated_frame = process_inference_results(
            selected_sperm,
            sperms_data,
            num_frame,
            results,
            frame,
            width,
            height,
            original_frame,
        )
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
