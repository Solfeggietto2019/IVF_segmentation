import cv2
from data_processing.json_reader import JSONReader
from data_processing.stream_reader import VideoStreamReader
from data_processing.yolo_model import YOLOModel
from data_processing.inference_logic import process_inference_results
from utils.utils import draw_positions, adjust_coordinates
from utils.dataclasses import SelectedSperm

def main() -> None:
    json_file_path = 'data/json/48115_Nadiya_0402221432.json'
    video_file_path = 'data/video/48115_Nadiya_0402221432.avi'
    model_path = 'models/best.pt'
    
    selected_sperm = SelectedSperm
    reader = JSONReader(json_file_path)
    video_reader = VideoStreamReader(video_file_path)
    yolo = YOLOModel(model_path)
    
    sperms_data = reader.extract_sperms_data()
    
    for video_data in video_reader:
        frame, width, height, fps, num_frame = video_data
        num_frame -= 1
        results = yolo.infer(frame)
        for sequence_id, sperm_list in sperms_data.items():
            for sperm in sperm_list:
                if sperm.initial_frame <= num_frame <= sperm.final_frame:
                    frame_coords = sperm.positions[num_frame - sperm.initial_frame]
                    x, y = frame_coords["posX"], frame_coords["posY"]

                    x_adjusted, y_adjusted = adjust_coordinates(x, y, frame.shape)
                    sperm.positions[num_frame - sperm.initial_frame]['posX'] = x_adjusted
                    sperm.positions[num_frame - sperm.initial_frame]['posY'] = y_adjusted
                    
                    frame = draw_positions(frame, (x_adjusted, y_adjusted), sperm.id)
            
        annotated_frame = process_inference_results(selected_sperm, sperms_data, num_frame, results, frame, width, height)
        cv2.imshow('YOLOv8 Tracking', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
