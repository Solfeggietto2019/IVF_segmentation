import cv2
from typing import Generator, Any


class VideoStreamReader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.file_path}")

    def __iter__(self) -> Generator[Any, None, None]:
        """
        Generator to read frames one by one from the video file.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            num_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"STARTING NUM FRAME", {num_frame})
            if not ret:
                break
            yield frame, width, height, fps, num_frame

    def release(self) -> None:
        """
        Release the video capture object.
        """
        self.cap.release()
