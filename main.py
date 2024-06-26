import cv2
from data_processing.json_reader import JSONReader
from data_processing.stream_reader import VideoStreamReader
from data_processing.yolo_model import YOLOModel
from data_processing.inference_logic import process_inference_results
from utils.utils import draw_positions, adjust_coordinates
from utils.dataclasses import SelectedSperm

def main() -> None:

    selected_sperm = SelectedSperm
    # Leer el archivo JSON
    json_file_path = 'data/json/48115_Nadiya_0402221432.json'
    reader = JSONReader(json_file_path)
    sperms_data = reader.extract_sperms_data()

    # Leer el archivo de video MP4 frame por frame y mostrar los resultados de YOLO
    video_file_path = 'data/video/48115_Nadiya_0402221432.avi'
    video_reader = VideoStreamReader(video_file_path)
    model_path = 'models/best.pt'  # Actualiza con la ruta real de tu modelo YOLO
    yolo = YOLOModel(model_path)

    for video_data in video_reader:
        frame, width, height, fps, num_frame = video_data
        num_frame -= 1
        # Ejecutar inferencia de YOLO en el frame
        results = yolo.infer(frame)
        # Iterar sobre los objetos Sperm en el frame actual
        for sequence_id, sperm_list in sperms_data.items():
            for sperm in sperm_list:
                if sperm.initial_frame <= num_frame <= sperm.final_frame:
                    frame_coords = sperm.positions[num_frame - sperm.initial_frame]
                    x, y = frame_coords["posX"], frame_coords["posY"]

                    x_adjusted, y_adjusted = adjust_coordinates(x, y, frame.shape)
                    sperm.positions[num_frame - sperm.initial_frame]['posX'] = x_adjusted
                    sperm.positions[num_frame - sperm.initial_frame]['posY'] = y_adjusted
                    
                    frame = draw_positions(frame, (x_adjusted, y_adjusted), sperm.id)
            
        # Procesar y mostrar los resultados de la inferencia
        annotated_frame = process_inference_results(selected_sperm, sperms_data, num_frame, results, frame, width, height)

        # Mostrar el frame anotado
        cv2.imshow('YOLOv8 Tracking', annotated_frame)
        # Esperar por la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
