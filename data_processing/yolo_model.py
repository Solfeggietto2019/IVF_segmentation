# yolo_model.py
import torch
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path: str) -> None:
        """
        Initializes the YOLOModel with the model path.

        :param model_path: Path to the YOLO model file.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print("USING GPU")
            torch.cuda.set_device(0)

        self.model = YOLO(model_path)

    def infer(self, frame):
        """
        Runs inference on a given frame using the YOLO model.

        :param frame: The frame on which to run inference.
        :return: The inference results.
        """
        #results = self.model.track(frame, persist=True, show_boxes=False, show_labels=True)
        results = self.model.predict(frame)
        return results

